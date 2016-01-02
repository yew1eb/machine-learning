# _*_ coding: utf-8 _*_

import re
import threading
from Queue import *
from urllib2 import *


class download(threading.Thread):
	def __init__(self, que):
		threading.Thread.__init__(self)
		self.que = que

	def run(self):
		while True:
			if not self.que.empty():
				print('-----%s------' % (self.name))
			#os.system('wget ' + self.que.get())
			else:
				break


def startDown(url, rule, num, start, end, decoding=None):
	"""

	:param url:
	:param rule:
	:param num:
	:param start:
	:param end:
	:param decoding:
	"""
	if not decoding:
		decoding = 'utf8'
	req = urlopen(url)
	body = req.read().decode(decoding)
	rule = re.compile(rule)
	link = rule.findall(body)
	que = Queue()
	for l in link:
		que.put(l[start:end])
	for i in range(num):
		d = download(que)
		d.start()


if __name__ == '__main__':
	url = 'https://class.coursera.org/algo-004/lecture/index'
	rule = '<a target=\"_new\" href=\".*\"'
	startDown(url, rule, 10, 23, -1)
