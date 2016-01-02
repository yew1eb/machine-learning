# _*_ coding: utf-8 _*_


import threading
import time


class timer(threading.Thread):
	#The timer class is derived from the class threading.Thread
	def __init__(self, num, interval):
		threading.Thread.__init__(self)
		self.thread_num = num
		self.interval = interval
		self.thread_stop = False

	def run(self):
	#Overwrite run() method, put what you want the thread do here
		while not self.thread_stop:
			print 'Thread Object(%d), Time:%s/n' % (self.thread_num, time.ctime())
			time.sleep(self.interval)

	def stop(self):
		self.thread_stop = True


def test():
	thread1 = timer(1, 1)
	thread2 = timer(2, 2)
	thread1.start()
	thread2.start()
	time.sleep(5)
	thread1.stop()
	thread2.stop()
	return


if __name__ == '__main__':
	test()
