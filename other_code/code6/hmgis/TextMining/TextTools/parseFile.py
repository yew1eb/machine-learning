# _*_ coding: utf-8 _*_

import re
import jieba.posseg as pseg
import jieba

# 将CSV格式的微博文本数据转化为分词后的文本数据
class parseCSV:
	def __init__(self, stopwords, dict):
		self.stopwords = stopwords
		self.dict = dict

	def parse(self, infile, outfile):
		## 读取微博数据
		weiboinfo = [line.strip() for line in file(infile)]
		## 选择Text部分进行处理
		sentences = [course.split(',')[1] for course in weiboinfo]
		## 剔除转发格式//@***:
		rgx = re.compile('(?<=//@).*?(?=:)')
		sentences = [rgx.sub('', sentence).replace('//@:', '') for sentence in sentences]
		## 剔除@**格式
		rgx = re.compile('(?<=@).*?(?= )')
		sentences = [rgx.sub('', sentence).replace('@', '') for sentence in sentences]
		## 剔除语气格式[哈哈]
		rgx = re.compile('(?<=\[).*?(?=\])')
		sentences = [rgx.sub('', sentence).replace('[]', '') for sentence in sentences]
		## 剔除回复
		rgx = re.compile('(?<=回复).*?(?=:)')
		sentences = [rgx.sub('', sentence).replace('回复:', '') for sentence in sentences]

		jieba.load_userdict(self.dict)
		## 使用jieba对文本进行分词
		texts_tokenized = [
			[word for word in pseg.cut(document) if word.flag in ['n', 'nr', 'ns', 'nt', 'nz', 'nl', 'ng', 'eng', 'x']]
			for document in sentences]
		## 停用词
		chnstopwordoc = [line.strip() for line in file(self.stopwords)]
		stoplist = [course for course in chnstopwordoc]
		## 剔除停用词
		texts_tokenized = [[word for word in document if not word.word in stoplist] for document in texts_tokenized]
		## 剔除长度为1的词汇
		wordlist = [[word.word for word in document if len(word.word) > 1] for document in texts_tokenized]

		OUT = outfile
		f = open(OUT, "w")
		str = ""
		for words in wordlist:
			for w in words:
				if len(w) > 0:
					str = str + w + "\t"
			if str != "":
				f.writelines(str + "\n")
			str = ""
		print "文件输出成功"

