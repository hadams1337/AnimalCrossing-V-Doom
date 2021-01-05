Animal Crossing or DOOM?
An exploration of reddit posts from each subreddit community
<p float="center">
  <img width="350" height="200" src=./cute/logos.jpg>
</p>

In this project I looked to machine learning to identify what the characteristics of each subreddit were and then build a model that can predict which subreddit a post came from.

Here are some example posts from each subreddit
<p float="left">
  <img width="350" height="300" src=./cute/acnh_reddit.png>
  <img width="350" height="300" src=./cute/doom_reddit.png>
</p>
First I processed the titles by removing stopwords and punctuation (other than ? and !) and using sklearn, TF-IDF(as opposed to a count of the words, this lowers the importance of the words that are most common and likely to appear in both, videogames, maybe.)
Then I performed Naive Bayes to find the words that were most likely to be present in Doom and Animal Crossing posts.  The size of the wedge shows the likelihood of each word being found in a post from the respective subreddit.



Titles- The size indicates the normalized likelihood that the word in the chart was in the title of the respective subreddit.
<p float="left">
  <img width="350" height="300" src=./graphs/acnh_titles_pie.png>
  <img width="350" height="300" src=./graphs/titles_doom_pie.png>
</p>
There are several important things to notice in this graph:

The distribution of the words for animal crossing are a lot more evenly sized, when people talk about doom, they use the same words most of the time.

The animal crossing words include the word doom!  That's likely because the games were released the same day, which gave rise to some crossover fan art. 


Body-More Natural Language Processing

Pictures -KNN to find 'mood' of the pictures

Processing to put that into a matrix

Naive Bayes:


Render unto Animal Crossing that which is Animal Crossing's
A computer was trained to identify titles, 'mood' of the picture, and body text of a subreddit post.











<p align="center"></p>
  <img width="430" height="300" src=./cute/animal_crossing_over_by_ry_spirit_ddqs9tg-fullview.jpg>
</p>

<p align="center"></p>
  <img width="430" height="300" src=./cute/doom_teach_isa.jpeg>
</p>