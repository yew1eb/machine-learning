# _*_ coding: utf-8 _*_
import logging
import jieba
from gensim import corpora, models, similarities


class GensimTest:
	def printTopicClass(self, corpus_text, corpus_lis):
		i = 0
		for corpus in corpus_lis:
			if len(corpus) == 0:
				print corpus_text[i], "属于 -1 主题"
			else:
				max = abs(corpus[0][1])
				tag = 0
				for t in corpus:
					if abs(t[1]) > max:
						max = abs(t[1])
						tag = t[0]
				print corpus_text[i], "属于", tag, "主题"
			i = i + 1

	def topicModelDict(self, corpus_text, corpus_lis):
		topic_dict = {}

		for (corpus, i) in zip(corpus_lis, range(0, len(corpus_lis))):
			if len(corpus) == 0:
				topic_dict[corpus_text[i]] = -1
			else:
				max = abs(corpus[0][1])
				tag = 0
				for t in corpus:
					if abs(t[1]) > max:
						max = abs(t[1])
						tag = t[0]
				topic_dict[corpus_text[i]] = tag

		info = sorted(topic_dict.items(), key=lambda d: d[1])
		return info

	def simple(self):
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

		## 给出一个文档
		documents = ["Human human machine interface for lab abc computer applications",
		             "A survey of user opinion of computer system response time",
		             "The EPS user interface management system",
		             "System and human system engineering testing of EPS",
		             "Relation of user perceived response time to error measurement",
		             "The generation of random binary unordered trees",
		             "The intersection graph of paths in trees",
		             "Graph minors IV Widths of trees and well quasi ordering",
		             "Graph minors A survey"]

		# 剔除停用词
		stoplist = set('for a of the and to in'.split())
		texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
		# 剔除仅出现过一次的词汇
		all_tokens = sum(texts, [])
		tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
		texts = [[word for word in text if word not in tokens_once] for text in texts]
		## 处理后的文档及其词汇组成
		print texts

		## 将文档添加到一个词典
		dictionary = corpora.Dictionary(texts)
		dictionary.save('data/LSA/deerwester.dict') # store the dictionary, for future reference
		##print dictionary
		## 显示单词及其ID
		print dictionary.token2id

		## 新建语料库corpus
		corpus = [dictionary.doc2bow(text) for text in texts]
		## 将语料库序列化备用
		corpora.MmCorpus.serialize('data/LSA/deerwester.mm', corpus) # store to disk, for later use
		## 现在corpus的格式如下，每个单词的id及其出现次数
		print "语料库在词典中的分布"
		print corpus

		tfidf = models.TfidfModel(corpus)
		corpus_tfidf = tfidf[corpus]
		lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
		corpus_lsi = lsi[corpus_tfidf]
		index = similarities.MatrixSimilarity(lsi[corpus])
		lsi.print_topics(2)

		print "所有文档的主题分布"
		for corpus in corpus_lsi:
			print corpus

		## 将新文本用字典来查找单词的ID
		new_doc = "Human computer interaction"
		new_vec = dictionary.doc2bow(new_doc.lower().split())
		## 看到interaction因为词典中不存在因此也不会出现
		print "新文档在词典中的位置"
		print new_vec # the word "interaction" does not appear in the dictionary and is ignored
		## 展示文档的所属的类型
		print "新文档的主题"
		query_lsi = lsi[new_vec]
		print query_lsi
		self.printTopicClass([new_doc], [query_lsi])

		# 与"Human computer interaction"前9位相似度的文档
		sims = index[query_lsi]
		sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
		print "与新文档最相关的文档"
		print sort_sims[0:9]


	def simple2(self):
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		texts = [line.lower().split() for line in open('lib/corpus.txt')]
		# collect statistics about all tokens
		dictionary = corpora.Dictionary(texts)
		# remove stop words and words that appear only once
		stoplist = set('for a of the and to in'.split())
		stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
		            if stopword in dictionary.token2id]
		once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
		dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
		dictionary.compactify() # remove gaps in id sequence after words that were removed
		print len(dictionary.token2id)

		corpus = [dictionary.doc2bow(text) for text in texts]
		## 新建基于TF-IDF权重模型的VSM
		tfidf = models.TfidfModel(corpus)
		corpus_tfidf = tfidf[corpus]

		##
		#print tfidf.dfs
		#print tfidf.idfs

		lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
		index = similarities.MatrixSimilarity(lsi[corpus])
		lsi.print_topics(2)

		query = "Human computer interaction"
		query_bow = dictionary.doc2bow(query.lower().split())
		print query_bow
		query_lsi = lsi[query_bow]
		print query_lsi
		self.printTopicClass([query], [query_lsi])

		# 与"Human computer interaction"前9位相似度的文档
		sims = index[query_lsi]
		sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
		print "与新文档最相关的文档"
		print sort_sims[0:9]

	## 中文分类
	def GIS3SNewsTopic(self):
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

		courses = [line.strip() for line in file('data/gensim/3snews')]
		courses_name = [course.split('\t')[0] for course in courses]
		for i in range(10):
			print courses_name[i]

		'''
		## 按词性分词对结果影响很大
		texts_tokenized = [
			[word.word for word in pseg.cut(document) if word.flag in ['n', 'nr', 'ns', 'nt', 'nz', 'nl', 'ng', 'eng', 'x']]
			for document in courses_name]
		'''
		texts_tokenized = [[word for word in jieba.cut(document, cut_all=False)] for document in courses_name]
		texts = [[word for word in document if len(word) > 1] for document in texts_tokenized]
		for t in texts:
			for t1 in t:
				print t1

		from gensim import corpora, models

		## 剔除所有只能出现了一次的特征性
		all_tokens = sum(texts, [])
		tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
		texts = [[word for word in text if word not in tokens_once] for text in texts]

		dictionary = corpora.Dictionary(texts)
		print len(dictionary.token2id)

		corpus = [dictionary.doc2bow(text) for text in texts]
		tfidf = models.TfidfModel(corpus)
		corpus_tfidf = tfidf[corpus]
		for s in corpus_tfidf:
			print s

		##这里我们拍脑门决定训练topic数量为5的LSI模型：
		lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=6)
		lsi.print_topics(6)

		corpus_lsi = lsi[corpus_tfidf]
		for corpus in corpus_lsi:
			print corpus

		dict = self.topicModelDict(courses_name, corpus_lsi)
		for b in dict:
			print str(b[0]) + " 属于主题 " + str(b[1])

	def weiboTopic(self):
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

		courses = [line.strip() for line in file('data/LSA/wb_clean.txt')]
		texts = [course.split('\t') for course in courses]

		from gensim import corpora, models

		## 剔除所有只能出现了一次的特征性
		all_tokens = sum(texts, [])
		tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
		texts = [[word for word in text if word not in tokens_once] for text in texts]

		dictionary = corpora.Dictionary(texts)
		print len(dictionary.token2id)

		corpus = [dictionary.doc2bow(text) for text in texts]
		tfidf = models.TfidfModel(corpus)
		corpus_tfidf = tfidf[corpus]
		for s in corpus_tfidf:
			print s

		##这里我们拍脑门决定训练topic数量为5的LSI模型：
		lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=6)
		lsi.print_topics(6)

		corpus_lsi = lsi[corpus_tfidf]
		for corpus in corpus_lsi:
			print corpus

		dict = self.topicModelDict(courses, corpus_lsi)
		for b in dict:
			print str(b[0]) + " 属于主题 " + str(b[1])
