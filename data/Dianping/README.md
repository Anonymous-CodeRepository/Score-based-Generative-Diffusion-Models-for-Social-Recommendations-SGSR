This is part of the anonymized Dianping (http://www.dianping.com) dataset used in our RecSys'15 paper for evaluating the quality of social recommender systems. It contains 147,918 users, 11,123 restaurants and 2,149,675 ratings from April 2003 to November 2013 in Shanghai, China. For the social friend network, there are a total of 629,618 claimed social relationships (undirected edge). For privacy issue, we do not include the user information and restaurant attributes which can be used to identify a real person.

User id and item id are consecutive numbers starting from 0.

- Format of "user.txt" file:

user id|ids of friends (separated by whitespace)

- Format of "rating.txt" file:

user id|item id|rating score|date

If you use the Dianping dataset, please cite our papers:


@inproceedings{LiWTM15,
  author    = {Hui Li and
               Dingming Wu and
               Wenbin Tang and
               Nikos Mamoulis},
  title     = {Overlapping Community Regularization for Rating Prediction in Social
               Recommender Systems},
  booktitle = {RecSys},
  pages     = {27--34},
  year      = {2015}
}