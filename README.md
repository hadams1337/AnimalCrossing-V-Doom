# Animal Crossing or DOOM?
An exploration of reddit posts from each subreddit community
<p float="center">
  <img width="350" height="200" src=./cute/logos.jpg>
</p>

In this project I looked to machine learning to identify what the characteristics of each subreddit were and then build a model that can predict which subreddit a post came from.

Here are some example posts from each subreddit
<p float="center">
  <img width="200" height="200" src=./cute/acnh_reddit.png>
  <img width="200" height="200" src=./cute/doom_reddit.png>
</p>

## Step 1 Naive-Cleaning and Naive-Bayes Classifications
First I processed the titles by removing stopwords and punctuation (other than ? and !) and using sklearn, TF-IDF(as opposed to a count of the words, this lowers the importance of the words that are most common and likely to appear in both, videogames, maybe.)
Then I performed Naive Bayes to find the words that were most likely to be present in Doom and Animal Crossing posts.  The size of the wedge shows the likelihood of each word being found in a post from the respective subreddit.



### Title Text- Average Length of Title - 6 words
The size indicates the normalized likelihood that the word in the chart was in the title of the respective subreddit.

<p float="left">
  <img width="250" height="250" src=./graphs/acnh_titles_pie.png>
  <img width="250" height="250" src=./graphs/titles_doom_pie.png>
</p>

There are several important things to notice in this graph:

The distribution of the words for animal crossing are a lot more evenly sized, when people talk about doom, they use the same words most of the time.

The animal crossing words include the word doom!  That's likely because the games were released the same day, which gave rise to some crossover fan art. 

### Body Text
Body-More Natural Language Processing
To fill out the picture of "what are common traits of doom and animal crossing posts  I scraped 1000 posts of each for body text and 
<p float="left">
  <img width="250" height="250" src=./graphs/a_t_b.png>
  <img width="250" height="250" src=./graphs/d_t_b_.png>
</p>
The presence of the words "know" and "poll" in both sets of posts likely means that people are asking questions like, "do you know how to..." or submitting polls for the majority of posts in both subreddits.

### Pictures -I used KNN to find 'mood' of the pictures
This step was slightly more complicated.  I was curious to see if I could use the colors of the pictures to see if the "mood" of the pictures varied.  I decided to use the relative prevalence of the color.

Processing to put that into a matrix

Naive Bayes:


# Step 2 Logistic Regression and Gradient Boosted Random Forests 
#### Render unto Animal Crossing that which is Animal Crossing's
A computer was trained to identify titles, 'mood' of the picture, and body text of a subreddit post.











<p align="center"></p>
  <img width="430" height="300" src=./cute/animal_crossing_over_by_ry_spirit_ddqs9tg-fullview.jpg>
</p>

<p align="center"></p>
  <img width="430" height="300" src=./cute/doom_teach_isa.jpeg>
</p>