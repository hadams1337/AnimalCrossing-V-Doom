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
I was curious to see if I could use the colors of the pictures to see if the "mood" of the pictures varied between subreddits.  I wanted to check two things: what are the color differences between the two subreddits, and how much accent color is there in each picture?  Does the number and importance of accent colors vary between the groups?
I average the colors for each picture using K-Nearest Neighbors to find 20 colors to represent the picture in RGB 256 values.  Then I rounded each value (red, green and blue) separately to the nearest 5 in hopes that I would get some repeat values, I didn't need precision to closer than 5 for each number.
The colors were recombined and formed the column names for a pandas dataframe in which the column names were the rgb value as a string and the values were the number of pixels in that file (row) that had that color.  This is analgous to a 'count vectorized' text file in which the 'word' is the RGB value and the 'count' is the number of pixels.  The total number of columns was 17,947, which is about half of what it would be if all 1600 pictures had distinct sets of 20 colors.  This gives me great hope that the pictures from each subreddit will be distinct from each other. The resultant dataframe is shown below.

<p float="center">
  <img width="550" height="200" src=./graphs/Color_words_df.png>
</p>

Once I had the count vectorized colors, I divided each value by 300x300 (the number of pixels total) to find the 'term frequency' of each color.  

Then I used SKLearn's Multinomial Naive Bayes to turn those frequencies into the probability that the color would be seen in each subreddit post.  Below, I have the final pie chart with the color, title, and body.  The colors are the inner most ring and the size of the each wedge is the probability that color will be present in the subreddit of choice.  The colors of each wedge are the actual colors.

<p float="left">
  <img width="250" height="250" src=./graphs/a_c_t_b.png>
  <img width="250" height="250" src=./graphs/d_c_t_b_.png>
</p>

The results were interesting!  Black was overwhelmingly likely to be present in Doom, while most of the Animal Crossing pictures had varieties of pink.  Doom pictures had about the same accent colors, blue and green with only small variations, and Animal Crossing had a variety of pink hued pastels with a small amount of green.



# Step 2 Logistic Regression and Gradient Boosted Random Forests 
#### Render unto Animal Crossing that which is Animal Crossing's
A computer was trained to identify titles, 'mood' of the picture, and body text of a subreddit post.











<p align="center"></p>
  <img width="430" height="300" src=./cute/animal_crossing_over_by_ry_spirit_ddqs9tg-fullview.jpg>
</p>

<p align="center"></p>
  <img width="430" height="300" src=./cute/doom_teach_isa.jpeg>
</p>