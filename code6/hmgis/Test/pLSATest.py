# _*_ coding: utf-8 _*_
import os
import glob
from hmgis.TextMining.pLSA.plsa import *


class pLSATest:
	def plsaTest(self):
		STOP_WORDS_SET = set()

		print "Usage: python ./main.py <number_of_topics> <maxiteration>"
		# load stop words list from file
		stopwordsfile = open("lib/stopwords.txt", "r")
		for word in stopwordsfile: # a stop word in each line
			word = word.replace("\n", '')
			word = word.replace("\r\n", '')
			STOP_WORDS_SET.add(word)

		corpus = Corpus() # instantiate corpus
		# iterate over the files in the directory.
		document_paths = ['data/plsa/texts/grimm_fairy_tales', 'data/plsa/texts/tech_blog_posts', 'data/plsa/texts/nyt']
		#document_paths = ['./test/']
		for document_path in document_paths:
			for document_file in glob.glob(os.path.join(document_path, '*.txt')):
				document = Document(document_file) # instantiate document
				document.split(STOP_WORDS_SET) # tokenize
				corpus.add_document(document) # push onto corpus documents list

		corpus.build_vocabulary()
		print "Vocabulary size:" + str(len(corpus.vocabulary))
		print "Number of documents:" + str(len(corpus.documents))

		number_of_topics = 5
		max_iterations = 5

		plsa = pLSA(corpus)
		plsa.plsa(number_of_topics, max_iterations)

		print plsa.document_topic_prob
		print plsa.topic_word_prob

		plsa.print_topic_word_distribution(corpus, number_of_topics, 20, "data/plsa/topic-word.txt")
		plsa.print_document_topic_distribution(corpus, number_of_topics, 5, "data/plsa/document-topic.txt")


