from scrapy.spiders import Spider
from lyrics_music.items import LyricsMusicItem
from scrapy.http    import Request
import re

#scrapy crawl lyricsspider  --set CLOSESPIDER_PAGECOUNT=5 -o items.json -t json

class LyricsSpider(Spider):
	name = "lyricsspider"
	allowed_domains = ["letras.mus.br"]
	start_urls = ["https://www.letras.mus.br/mais-acessadas/mpb/"]

	def make_requests_from_url(self, url):
		return Request(url, dont_filter=True, meta = {'start_url': url})

	def parse(self, response):
		links = response.xpath("//ol[@class='top-list_mus cnt-list--col1-2']/li/a/@href").extract()

		#stored already crawled links in this list
		crawledLinks = []

		#pattern to check proper link
		linkPattern = re.compile("^[a-z+-?a-z]+\/\d+")

		for link in links:
			#get proper link and is not checked yet
			if linkPattern.match(link) and not link in crawledLinks:
				link = "https://www.letras.mus.br" + link
				crawledLinks.append(link)
				yield Request(link, self.parse)

		titles = response.xpath("//div[@class='cnt-head_title']")
		item = LyricsMusicItem()
		if(len(titles.xpath("//h1/text()").extract()) > 0):
			item["title"] = titles.xpath("//h1/text()").extract()[0]
			item["lyrics"] = ' '.join(titles.select("//article/p/text()").extract())
			item["genre"] = titles.xpath("//div[@id='breadcrumb']/span/a/span/text()").extract()[1]
			item["link"] = response.url			
			yield item
