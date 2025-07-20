## Sentiment Analysis of Bsky Posts

In this Project, we want to classify the sentiment (positive, negative, neutral)
of [Bsky](https://bsky.app/) posts.
To this end, we first train a model on labeled tweets using the [gxb912/large-twitter-tweets-sentiment](https://huggingface.co/datasets/gxb912/large-twitter-tweets-sentiment)
dataset. However, we do not use the provided labels and instead use a *teacher* model (a LLM) to label the data; 
we distill the LLM into a smaller, specialized language model for sentiment analysis.
This allows us to harness the predictive power of a LLM with significantly lower hardware requirements. 

Additionally, we deploy the model as part of a fullstack app which allows a user to provide a Bsky feed
(their own or a global *hot* feed) to infer the general sentiment of the posts in the feed.

### Project Structure

* `data` Contains data for training the sentiment prediction model
* `sentiment-analyis` Contains notebooks and scripts for analyzing posts, data preparation, model training and evaluation
* `models` Checkpoints of the sentiment prediction model
* `app` Front- and Backend for analyzing Bsky feeds
