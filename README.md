Projeto de Classificação de Letras de Músicas em Gêneros Musicais
Desenvolvido por Ana França e João Pedro Magalhães em Python 2.7.
Usa as bibliotecas NLTK, Scikit-learn, langdetect e Scrapy.

Arquivos:
metrics_file.json  -  resultados dos experimentos antes dos ajustes
metrics_file_tunning_new  -  resultados dos experimentos após os ajustes
analyzer.py  -  programa para ajudar na análise de resultados. Dado um json de metricas, pode se fixar parametros para filtrar e ordenar os resultados
analysis.txt  -  arquivo de saída do analyzer.py
count_artists.py  -  conta o número de artistas contidos nas músicas usadas
classification.py  -  programa base do projeto.

Pastas:
lyrics_music  -  onde se encontra todo o corpus
