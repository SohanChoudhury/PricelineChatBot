
# Priceline Chatbot Schema
### by Sohan Choudhury
> Applicant for the role of [Machine Learning Intern](https://careers.priceline.com/jobsearch/job-details/summer-intern-machine-learning-engineer/oMN16fwq/1/), Summer 2018

___

This is a methodology for the creation of a customer service chatbot, specifically for implementation by [Priceline](https://www.priceline.com). Below I discuss my interest in this role, features for exploration, as well as an in depth explanation of relevant NLU and ML architectures.

## Motivations

In order to initially spur my interest, as well as to demonstrate my commitment to this role, I created my own chatbot using Microsoft Azure's AI as a proof of concept. Although a preliminary tool, the bot was able to accurately determine whether or not the customer intended to book a hotel, car, flight, or cruise using Priceline. Below is a video demonstration.

<!-- <a href="http://www.youtube.com/watch?feature=player_embedded&v=JZ5uHs8UQAY
" target="_blank"><img src="http://img.youtube.com/vi/JZ5uHs8UQAY/0.jpg"
alt="Bot 1.0 Demo" width="600" height="450" border="10"></a> -->

[![Priceline Chatbot Demo](http://img.youtube.com/vi/JZ5uHs8UQAY/0.jpg)](http://www.youtube.com/watch?v=JZ5uHs8UQAY "Priceline Chatbot Demo")


> Click the thumbnail to view the demo.

Like most chatbots in the travel industry, this bot utilizes supervised learning. I hardcoded desired outputs, as well as some example inputs which I used to train it. The goal was for the system to learn general rules governing mapping inputs to outputs. Here, the support vector machine (SVM) was provided by Microsoft Azure.



## Wrangling User Data

In order for the chatbot to function properly, the most obvious input stream we have is the information the user provides. This, of course, is in the form of natural language sent to the chatbot in conversation. In order to provide appropriate responses, we must turn this language into data with text analytics.

Prior to word vectorization, however, we must design some sort of processing in order to extract the data from the user's messages, as well as to ensure that only relevant words are considered.

To this end, `nltk`, a popular Natural Language Toolkit for `Python`, can be used. We can use this handily remove common **_stopwords_** such as *the*, *and*, *so*, etc. Below is a simple function outlining how we can strip these words from our data.

```python
from nltk.corpus import stopwords

def clean(words):

  filtered_words = []

  for word in words:
    if word not in stopwords.words('english')
    filtered_words.append(word)

  return filtered_words
```
In order to make our processing even more robust, we can calculate the frequency at which the customer mentions certain phrases. This analysis can be done using `nltk` as well, given that we initially parse through `words`, assigning values corresponding with level of relation to certain predefined topics.

 For instance, if within the conversation words related to cost are mentioned at a high rate, it may be advisable to promote discounts or lower fares. We can assign each word a `cost_score`, taken from a publicly available dataset, and use this to score the overall importance of this factor to the customer.

```python
from nltk.probability import FreqDist

def cost_score(filtered_words, words):

  total_freq = 0.0
  ave_cost_importance = 0.0

  freq = FreqDist(filtered_words)
  total_freq = sum([freq[word] for word in filtered_words])

  for word in words:
    cost_score = public_data_set.get(word, 0)
    norm_freq = freq[word] / total_freq
    ave_cost_importance += cost_score * norm_freq

    return ave_cost_importance
```

From my understanding, the NLP engine has already been applied to the Priceline chatbot, such that it can handle various customer questions. In order to gather classification data, a word2vec model can be used to produce word embeddings. Word2vec can then use the skip-gram architecture to essentially predict the context of a user's input, and thereby their intent, from a given word. In practice, this gives us the following key benefit.

* **Named-entity recognition.** This aids in classifying named entities in text into predefined categories. This is particularly useful for determining what Priceline service a user may be interested in.
  * The unannotated block of text `My sister-in-law wants to fly out of JFK on April 6th.`
  * Becomes `My [sister-in-law]`<sub>`Person`</sub>`wants to fly out of [JFK]`<sub>`Airport`</sub>`on [April 6th]`<sub>`Date`</sub>`.`
  * Which is stored for later processing as:
  <table>
    <thead>
      <tr>
        <th>Person</th>
        <th>Airport</th>
        <th>Date</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>"sister-in-law"</td>
        <td>`JFK`</td>
        <td>`2018-04-06`</td>
      </tr>

    </tbody>
  </table>


There are various word2vec models that can be used for this. One example comes from [TensorFlow](https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/examples/tutorials/word2vec/word2vec_basic.py). We can use a relevant training set, [such as this one](http://download.tensorflow.org/data/questions-words.txt), also provided by TensorFlow, which can vectorize the relationships between the words coming in from the user, as shown below.

![picture alt](https://www.tensorflow.org/images/linear-relationships.png "Word2vec Representation")

With word2vec, we can effectively enable the NLU system to perform named-entity recognition, normalized for verb tense and accounting for simple relationships and hierarchies.

Now, we can store this information into a `pandas` dataframe in order to manipulate and prepare it for use by the chatbot. Below is a sampling of a few of the data points we may be able to collect. Each represent a column in the dataframe.

- `name` String representation of the customer's name

- `psize` Integer value of party size

- `budget` Float value of customer's indicated budget

- `start` String representation of booking start date in ISO format

- `end` String representation of booking start date in ISO format

- `origin` String representation of origin location of customer

- `destination` String representation of destination location of customer

- `car`Boolean value indicating if customer wants to book a vehicle

- `hotel` Boolean value indicating if customer wants to book a room

- `flight` Boolean value indicating if customer wants to book a flight

- `cruise` Boolean value indicating if customer wants to book a cruise


This data can now directly be used to generate responses by the chatbot. Although I am not familiar with the exact architecture of the current chatbot, I read that it utilizes JavaScript in the form of Node.js as well as Socket.IO.

For my aforementioned attempt using Microsoft Azure, I had used a JavaScript backend to communicate with the customer. For example, when a user input was identified to match `hotel`, a prepared response was randomly chosen and sent to the chat session on behalf of the bot.

```javascript

bot.dialog(`HotelDialogue`,

  (session) => {

    var possibleResponses = [
      "Want to book a hotel? Head over to https://www.priceline.com/hotels/."
      "If you want to book a hotel, head over to https://www.priceline.com/hotels/."
      "Looking for a room? Try https://www.priceline.com/hotels/."
    ];

    var response = possibleResponses[Math.floor(Math.random() * possibleResponses.length)];
    session.send(response);
  }

).triggerAction({
  matches: 'hotel'
})

```

I'm interested to see exactly how the processing is done on the current Priceline chatbot. With layered functionality and flexibility in responses, ML can be used to improve the chatbot's effectiveness over time.

## Implementing Machine Learning

In any prediction problem, there are several approaches that can be taken to build the classifier. A relevant and potentially fruitful approach here would be *ensemble learning*, which involves the combination of several models in order to solve a single prediction model.

*Random forest* is a type of ensemble learning commonly used for *feature selection*. With machine learning, we want to determine which features, or variables, are most significant. We can create many random decision trees, which will ultimately give us the random forest's overall prediction.

In regards to such a chatbot, the goal will be to identify the nature of responses that best improve effectiveness. At the crux of this, however, lies the problem of finding an adequate metric for judging effectiveness. I brainstormed a few, shown below.

* __Duration of conversation.__ It would be easy to measure how long a conversation lasted, and flag longer interactions as a sign of success. However, this simply is not an accurate metric for effective communication simply by the nature of conversation.

* __Explicit customer feedback.__ At the end of interaction with the chatbot, the user could be asked to rate the quality of their conversation. Unfortunately, this approach is highly susceptible to response bias and is quite subjective.

* __Implicit customer feedback.__ Sentiment analysis could be performed on the entirety of the user's sent messages, upon termination of the conversation.

This third option, utilizing sentiment analysis, has many benefits. It addresses concerns of response bias, and is a more representative metric of the success of a conversation. In order to gauge the effectiveness of the chatbot, we can analyze the sentiment score of the user after a conversation.

For this, we can use [LabMT](http://journals.plos.org/plosone/article/file?id=info%3Adoi/10.1371/journal.pone.0026752.s001&type=supplementary), a scored word list dataset for language assessment, to retrieve the *average happiness* of a word. The dataset contains 10,222 entries, and is in the following format.

<table>
  <thead>
    <tr>
      <th>`word`</th>
      <th>`happiness_rank`</th>
      <th>`happiness_average`</th>
      <th>`happiness_standard_deviation`</th>
    </tr>
  </thead>
  <tfoot>
    <tr>
      <td>`terrible`</td>
      <td>`9902`</td>
      <td>`2.84`</td>
      <td>`1.8111`</td>
    </tr>
  </tfoot>
  <tbody>
    <tr>
      <td>`thanks`</td>
      <td>`245`</td>
      <td>`7.40`</td>
      <td>`1.5119`</td>
    </tr>
    <tr>
      <td>`nice`</td>
      <td>`258`</td>
      <td>`7.38`</td>
      <td>`1.5104`</td>
    </tr>
    <tr>
      <td>`helpful`</td>
      <td>`364`</td>
      <td>`7.24`</td>
      <td>`1.2382`</td>
    </tr>
    <tr>
      <td>`weird`</td>
      <td>`9001`</td>
      <td>`4.20`</td>
      <td>`1.3553`</td>
    </tr>
  </tbody>
</table>

This information can be loaded into a `pandas` dataframe, after which we can convert the `happiness_average` series into a dictionary for quick reference when scoring sentiment of user responses.

```python
import pandas as pd

labmt_url = #LabMT URL linked above
labmt = pd.read_csv(url, skiprows=2, sep='\t', index_col=0)
labmt_dict = lambt.happiness_average.to_dict()

```

Now, we can take the ordered list of `words` taken from the user input, and combine the outline of both the `clean()` and `cost_score()` functions above to create `sentiment()` that gives us an average sentiment score for each conversation with the chatbot.

```python
from nltk.corpus import stopwords
from nltk.probability import FreqDist

def sentiment(words):

  total_freq = 0.0
  ave_sentiment = 0.0

  filtered_words = [word for word in words if word not in stopwords.words('english')]
  freq = FreqDist(filtered_words)
  total_freq = sum([freq[word] for word in filtered_words])

  for word in words:
    sentiment_score = labmt_dict.get(word, 0)
    norm_freq = freq[word] / total_freq
    ave_sentiment += sentiment_score * norm_freq

  return ave_sentiment

```

Based off of the scoring metric used in the dataset, the calculated `ave_sentiment` for a customer will be between `1.00` and `9.00`. If we take the running average of all customer `ave_sentiment` scores, we can then flag each conversation with the boolean tag `effective`, `True` indicating if the conversation was one or more standard deviations above the mean for sentiment, `False` if it was one or more standard deviations below. This effectively tells us which conversations were particularly effective, as well as which ones were substandard.

With a solid metric by which to identify effective and ineffective conversations, we can now move on to the task of feature engineering, where we construct features for insightful predictions.

As I am not familiar with the current chatbot and its features or response architecture, below are features I've arbitrarily chosen for the purpose of demonstration.

* **Discount** binary flag determining whether or not a discount was offered by the bot

* **Recommendation** binary flag determining whether or not a recommendation for another service was offered

* **No_Answer** binary flag determining whether or not the bot failed to give an answer

* **Personalization** binary flag determining whether or not the bot called the customer by `name`

After some feature manipulation, we should get a `pandas` dataframe that looks like this.

<table>
  <thead>
    <tr>
      <th></th>
      <th>**Discount**</th>
      <th>**Recommendation**</th>
      <th>**No_Answer**</th>
      <th>**Personalization**</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>`0`</td>
      <td>`1`</td>
      <td>`0`</td>
      <td>`0`</td>
      <td>`0`</td>
    </tr>
    <tr>
      <td>`1`</td>
      <td>`1`</td>
      <td>`1`</td>
      <td>`0`</td>
      <td>`0`</td>
    </tr>
    <tr>
      <td>`2`</td>
      <td>`1`</td>
      <td>`0`</td>
      <td>`1`</td>
      <td>`0`</td>
    </tr>
    <tr>
      <td>`3`</td>
      <td>`1`</td>
      <td>`0`</td>
      <td>`0`</td>
      <td>`1`</td>
    </tr>
    <tr>
      <td>`4`</td>
      <td>`1`</td>
      <td>`1`</td>
      <td>`1`</td>
      <td>`0`</td>
    </tr>
    <tr>
      <td>`5`</td>
      <td>`1`</td>
      <td>`0`</td>
      <td>`1`</td>
      <td>`1`</td>
    </tr>
    <tr>
      <td>`6`</td>
      <td>`1`</td>
      <td>`1`</td>
      <td>`1`</td>
      <td>`1`</td>
    </tr>
    <tr>
      <td>**...**</td>
      <td>**...**</td>
      <td>**...**</td>
      <td>**...**</td>
      <td>**...**</td>
    </tr>
  </tbody>
</table>

Following this pattern, as none of the features are mutually exclusive, we will have a total of 16 such rows, spanning `0` - `15`.

Now, the aforementioned random forest classifier can be trained. Here, the two inputs will be the training input samples (from the above dataframe), and the target values. As we want to identify the features that best promote positive user sentiment above a certain threshold, our target values will be `True` or `False`, as explained above.

Scikit-learn has a random forest classifier, `sklearn.ensemble.RandomForestClassifier`, that we can use for this task.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(train_x, train_y)

# where train_x is the features dataframe
# where train_y is the target sentiment boolean value
```

Now, we can rank which features were most important, and adjust the schematics of the chatbot accordingly. The following values are randomly generated for the purpose of demonstration.

```python
sorted(list(zip(classifier.feature_importances_, train_x.columns.tolist())),
		reverse=True)

>> [(0.39226792518662743, 'Discount'),
 	   (0.20912367099165802, 'Recommendation'),
 	   (0.18058298031704362, 'No_Answer'),
 	   (0.11673747109412265, 'Personalization'),]
```
From this, we can intuitively note that the features with the highest rankings are of the most importance in relation to the success of the chatbot, as measured by user sentiment. Using this information, we can then manipulate the schematics of the chatbot itself to improve results over time.

```python
rf.score(train_x, train_y)

>> 0.760325476992
# randomly generated value
```

As we gather more information from customer conversations and improve the chatbot, we can continuously gauge the accuracy of our model in relating features to success using the command above.

This concludes my take on the schematics for such a bot. Using NLU and ML, an incredibly versatile and insightful chatbot can be created and improved.

-----

Thank you for your time and consideration.
