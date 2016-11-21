from lyrics_music.items import LyricsMusicItem
from scrapy.spiders import Spider
from scrapy.http import Request
from langdetect import detect
import json, re, os

#scrapy crawl lyricsspider  --set CLOSESPIDER_PAGECOUNT=5 

class LyricsSpider(Spider):

	listUrl = []
	with open(os.path.abspath("artistspider_output.json"), 'r') as f:
		reader = json.load(f)
		listUrl = [links['link'] for links in list(reader)[1:] if len(links) > 0]

	name = "lyricsspider"
	allowed_domains = ["letras.mus.br"]
	start_urls = listUrl

	def parse(self, response):
		links = response.xpath("//div[@class='cnt-list--alp']/ul/li/a/@href").extract()

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
		if(len(titles.select("//article/p/text()").extract()) > 0):
			isPortuguese = (detect('\n '.join(titles.select("//article/p/text()").extract())) == 'pt')
			if(isPortuguese): #verify lyrics language before add in file
				item["title"] = titles.xpath("//h1/text()").extract()[0]
				item["lyrics"] = '\n '.join(titles.select("//article/p/text()").extract())
				item["genre"] = titles.xpath("//div[@id='breadcrumb']/span/a/span/text()").extract()[1]
				item["link"] = response.url			
				yield item
