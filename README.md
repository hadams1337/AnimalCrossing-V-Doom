# Animal Crossing or DOOM?
An exploration of reddit posts from each subreddit community
<p float="middle">
  <img width="350" height="200" src=./cute/logos.jpg>
</p>

In this project I looked to machine learning to identify what the characteristics of each subreddit were and then build a model that can predict which subreddit a post came from.

Here are some example posts from each subreddit
<p float="center">
  <img width="250" height="250" src=./cute/acnh_reddit.png>
  <img width="250" height="250" src=./cute/doom_reddit.png>
</p>

# Step 1 Text-Cleaning and Naive-Bayes Classifications

First I processed the titles by removing stopwords and punctuation (other than ? and !, which I thought might indicate mood), the words were stemmed and tokenized and TF-IDF was performed.
Then I performed Naive Bayes to find the words that were most likely to be present in Doom and Animal Crossing posts.  


## Title Text- Average Length of Title - 5 words
Below, pie charts are presented for each subreddit.  The size of the wedge shows the likelihood of each word being found in a post from the respective subreddit.

<p float="left">
  <img width="250" height="250" src=./graphs/acnh_titles_pie.png>
  <img width="250" height="250" src=./graphs/titles_doom_pie.png>
</p>

There are several important things to notice in this graph:

The distribution of the words for animal crossing are a lot more evenly sized, when people talk about doom, they use the same four words (doom, eternal, slayer, and like) most of the time.

The animal crossing words were more evenly distributed, and they include the word doom!  That's likely because the games were released the same day, which gave rise to some crossover fan art. 

## Body Text -More Natural Language Processing

To fill out the picture of "what are common traits of doom and animal crossing posts",  I scraped 1000 posts of each for body text.  The body text words were added to the title text pie charts.  I did this because I like the idea that we're adding together pieces of information to make a picture of the post.  The size of the wedge indicates how likely the word was to be found in each subreddit.


<p float="left">
  <img width="250" height="250" src=./graphs/a_t_b.png>
  <img width="250" height="250" src=./graphs/d_t_b_.png>
</p>
The presence of the words "know" and "poll" in both sets of posts likely means that people are asking questions like, "do you know how to..." or submitting polls for many posts in both subreddits.

## Pictures -KNN to find 'mood' of the pictures
I was curious to see if I could use the colors of the pictures to see if the "mood" of the pictures varied between subreddits.  I wanted to check two things: what are the color differences between the two subreddits, and how much accent color is there in each picture?  Does the number and importance of accent colors vary between the groups?
I average the colors for each picture using K-Nearest Neighbors to find 20 colors to represent the picture in RGB 256 values.  Then I rounded each value (red, green and blue) separately to the nearest 5 in hopes that I would get some repeat values, I didn't need precision to closer than 5 for each number.
The colors were recombined and formed the column names for a pandas dataframe in which the column names were the rgb value as a string and the values were the number of pixels in that file (row) that had that color.  This is analgous to a 'count vectorized' text file in which the 'word' is the RGB value and the 'count' is the number of pixels.  The total number of columns was 17,947, which is about half of what it would be if all 1600 pictures had distinct sets of 20 colors.  This gives me great hope that the pictures from each subreddit will be distinct from each other. The resultant dataframe is shown below.

<p float="center">
  <img width="550" height="250" src=./graphs/Color_words_df.png>
</p>

Once I had the count vectorized colors, I divided each value by 300x300 (the number of pixels total) to find the 'term frequency' of each color.  

Then I used SKLearn's Multinomial Naive Bayes to turn those frequencies into the probability that the color would be seen in each subreddit post. Below, I have the final pie chart with the color, title, and body (the full picture of a post).  The colors are the inner most ring and the size of each wedge is the probability that color will be present in the subreddit of choice.  The colors of each wedge are the KNN colors.

<p float="left">
  <img width="250" height="250" src=./graphs/a_c_t_b.png>
  <img width="250" height="250" src=./graphs/d_c_t_b_.png>
</p>

The results were interesting, black was overwhelmingly likely to be present in Doom, while most of the Animal Crossing pictures had varieties of pink.  Doom pictures had less varienty in accent colors (blue and green with only small variations) and Animal Crossing had a variety of pink hued pastels with a small amount of green.


# Step 2 Classification 
## Render unto Animal Crossing that which is Animal Crossing's

The 20 averaged colors above were further broken down into the number of pixels, red, green and blue values for each color.  This gave 80 columns containing the pixel count of the most prominent color to least prominent colors and respective red, green, and blue values from 0 to 255.  The dataframe is shown below.  Using this approach distills the 900,000 pixels with 3 colors each (2,700,000 pieces of information) to just 80 values per image.  I did this mostly because I wanted to see how abstract I could make the picture and still have it be somewhat recognizable to classification.
Those colors, titles, and body text were classified using logistic regression, random forest, and gradient boosted random forest.  The results are shown in the table below. 

<p float="center">
  <img width="550" height="250" src=./graphs/colors.png>
</p>

<table> 
<tr>
<th columnspan=5>Accuracy Data for Model Types</th>
</tr>
<th>Data</th><th>Logistic Regression
(color)</th><th>Random Forest
(color)</th><th>Gradient Boost
(color)</th><th>Logistic Regression
(body)</th>
<tr>
<td>Colors/Body</td>
<td>0.68</td><td>0.75</td><td>0.76</td><td>0.88</td>
</tr>
<tr>
<td>Title</td>
<td>0.71</td><td>0.71</td><td>0.73</td><td>0.81</td>
</tr>
</table>

The body text (unsurprisingly) has the best accuracy, near 90%.  This is likely due to the larger number of words in the average body text.  It is remarkable that the title (of posts with body text rather than pictures) had 80% accuracy given an average length of 5 words, but the Naive-Bayes provides some understanding since the name of the game frequently appeared in the titles of each game. Titles of picture posts peaked at 73% accuracy.
Misclassified titles tended to be short, for example, "Black Lives Matter" was a title to an Animal Crossing post but was classified as Doom.  Titles in which a person claimed to have made something tended to be classified as Animal Crossing.  
The surprising story was that 76% of the time the gradient boosted random forest could take the 80 columns of data for the colors and produce the correct classification.
The ROC curves for each classification are presented below.

<p float="center">
  <img align="middle" width="650" height="250" src=./graphs/ROC_curves.png>
</p>
The ROC curve for body words (left graph) have a relatively smooth curve, although even at low values of true positives, some false positives are present. 
The color values ROC curve (middle graph) is interesting, showing a distinct plateau.  This indicates that there is a type or types of pictures that the model is unlikely to classifiy as animal crossing until the threshold is moved to nearly 100% false positives.  Investigation showed that these pictures were nearly all black or all white, and tended to be twitter screenshots.  This falls in line with the Naive-Bayes color story of black being likely to be in doom posts.  Additionaly, the accent colors played a much smaller role for doom posts, the major color tended to fill up more pixels, which is likely what led to white backgrounds being classified as doom.
Finally, the  title text ROC curve (right graph) has nearly vertical slope initially, likely due to the high likelihood of finding the name of the game in the title. 

# Conclusion
The doom and animal crossing subreddits were distinctly different in text and 'mood'.  Very little information was needed to make a reasonable guess on the subreddit identity of a post.  The titles were lowest accuracy at 73%, but this is impressive considering that the average length of the title was 5 words.  The colors allowed 76% accuracy, also impressive given the reduction of information from over 2 million pieces per image to 80.  Finally, the body text had 88% accuracy.

# Next Steps
I'd like to combine each section to get a total score.  Here are some ideas for how that could look:
1) Use sentiment analysis to find the most common sentiment in each subreddit and add that to color classification dataframe
2) Use google's NIMA (neural image assessment) to assign an aesthetic score to each image and use that to enhance the color dataframe classification
3) Use a CNN to classify images or google's image text analysis to determinet text, and use TF-IDF on image text added to title text



<p align="middle"></p>
  <img width="200" height="200" src=./cute/animal_crossing_over_by_ry_spirit_ddqs9tg-fullview.jpg>
  <img width="200" height="200" src=./cute/doom_teach_isa.jpeg>
</p>