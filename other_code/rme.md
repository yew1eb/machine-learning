This repo is about machine learning.
TOOLS:
----->roc:roc.py
      get the point of roc
----->model_evaluate.py
      report the precision, recall and accuarcy. 

Classification:
----->DecisionTree : decision_tree.py
      http://en.wikipedia.org/wiki/Decision_tree 
      For detail about how to build a decision tree. 
      also, if you can read chinese, this aritcal may be helpful to you. 
      http://zengkui.blog.163.com/blog/static/2123000822012103113739819/
----->LogisticRegression
      More information with chinese 
      http://zengkui.blog.163.com/blog/static/21230008220127111040747/
----->LinearRegression
      More detail information with chinese
      http://zengkui.blog.163.com/blog/static/21230008220127411250679/
----->K-NN classification algorithm. 
      Data is represented by vsm and the features in vsm are the words 
      selected by max info gain. The distance between samples is measured 
      by cosine.
----->Naive Bayes 
      Naive Bayes classification algorithm.
      Bayes formular : P(C|w) = P(C,w) / P(w) = P(w|C) * P(C) / P(w)
      C : the category of articel.
      w : the words in articel
      Suppose that the probability of occurrence of a word in an article is 
      independent. We can classify the article by the following formular:
      P(C_i|w_1,w_2...w_n) = P(w_1,w_2...w_n|C_i) * P(C_i) / P(w_1,w_2...w_n)
      = P(w_1|C_i) * P(w_2|C_i)...P(w_n|C_i) * P(C_i) / (P(w_1) * P(w_2) ...P(w_n))
----->Adaboost
      Adaboost classification algorithm.
      you can get more detail info with the following link:
      http://en.wikipedia.org/wiki/AdaBoost
      http://zengkui.blog.163.com/blog/static/21230008220121110111925175/
Cluster:
----->K-means
      More detail information with chinese
      http://zengkui.blog.163.com/blog/static/2123000822012101784440471/


 
