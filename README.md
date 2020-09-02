# Cuisine Predictor

The main aim of this project is to develop a analyzer which takes the several json files of recipes and predicts the cuisine of the inputted ingredients. The project also provides closest cuisines possible for the given list of ingredients.

High-level overview.
1) Takes recipe json files which are around 38 thousand.
2) Generates embeddings for the input and ingredients of the 38 thousand files.
3) Finally predicts the cuisine type for the input ingredients provided and also gives the top-N foods which are made up of same ingredients.

The step-wise functionality of achieving the objectives.
1) Read the data of recipe json files and store them into a dataframe.
2) Ask the user to input all the ingredients that they are interested in.
3)Pre-train necessary classifier using the embeddings generated from the existing dataset.
4)Use the model to predict the type of cuisine and display it to the user.
5)Find the top N closest foods (you can define N). Return the IDs of those dishes to the user.

### Author - Nithivarn Reddy Shanigaram 

### Email - nithivarn.reddy.shanigaram-1@ou.edu

## External Packages used 

> pandas

> numpy

> scikit-learn


## Steps to install the project

This is a .ipynb so you can download it and run it on your jupyter or upload it your google colab and then run each cell.
When running the Analyzer.ipynb file in jupyter if it can't find the packages, then install them using conda installer , if you are using anaconda python development package.
If you are just using jupyter notebook then install the required packages using pip3 installer.

## Steps to Run the project
First download the recipe json files using this link (https://www.dropbox.com/s/f0tduqyvgfuin3l/yummly.json).
Next download the Analyzer.ipynb and get the path where your json data is downloaded or where it is placed.
Provide the path as a string to read_json_df() method.
Later Provide the input containing ingredients and then run each cell, there after.

## Assumptions made in the project are

1) The ingredients present in the dataset need not be normalized as it is only up of spices or other food items and I assume that normalizing the ingredients doesn't have much significance.

2) Also , as the json data doesn't have any NA or null values , I am not handling it. I have confirmed that the dataset has all field values. 

3) I have used RandomforestClassifier as my predictor after comparing the accuracies of several classifiers on the dataset.

4) I know that we can't perfectly say that which classifier is best using train_test_split technique on the dataset, but over here I have only used that metric in calculating the accuracies of several classifiers and then selecting that classifier.

5) Only one input list is provided by the user and it is comprised of strings.

### Functionality of each method in .ipynb

#### read_json_df(path=None):
This method reads the json data from file provided by the path and returns a dataframe which is used by the subsequent methods.
    
#### generate_tf_idf(df,input_ingredient):
This method takes the dataframe containing the json data and converts the ingredients list of each recipe into a string and appends it to a ingredients_corpus list. We can also convert the input_ingredient list provided by the user to a string and appended it as the last item to the ingretdients_corpus list . Now we pass the list to a Tfidfvectorizer and get the feature matrix.

#### prediction(df,doc_matrix):
This method takes the json data df, tfidfvectorizer generated feature_matrix and then we filter the doc_matrix to take all the feature_rows except the final row and also take the cuisine labels of corresponding rows from the dataframe and pass it to the randomforestclassifier model to train it supervisedly. Then we take the last row which is the input feature vector and pass it to the model to predict the cuisine label. Once the cuisine label is predicted it is returned.

#### top_n_similar(doc_matrix,df,n_top=8):
This method takes the feature matrix , dataframe and the n_top value which is the number of top dishes we want to display.
I have used cosine_similarity method from sklearn to get all the similarity scores between the input feature vector and the all the other feature vectors of the recipe files. Then I sorted the array of scores in reverse order and taken only the n_top indices from them. Then using those indices I have traversed through the list of ids , list of cuisines obtained from the dataframe and formed a list of tuples containing the recipe_id , score , cuisine type. This method returns the list containing the top-N foods.

References used - Textbook and Sklearn package.

#### Output

The output is a predicted cuisine type for the input ingredients and also the top_n closest foods along with the scores of relation is outputted. The closest cuisine list is made of tuples each in the following format (id,score,cuisine). "id" is the id of the corresponding recipe,"score" generated from cosine_similarity,"cuisine" of the corresponding recipe.

##### Sample Output
Input Ingredients -  ['chili powder', 'pepper', 'butter', 'bread', 'chicken', 'lettuce']
Predicted cuisine type for above ingredients - ['mexican']
Closest 8 cuisines with scores (id,score,cuisine)- [(33393, 0.4467, 'mexican'), (19277, 0.4421, 'russian'), (42954, 0.4191, 'mexican'), (15753, 0.4139, 'russian'), (34743, 0.4015, 'thai'), (31220, 0.3963, 'mexican'), (13975, 0.3945, 'french'), (37087, 0.3857, 'indian')]

### For questions regarding the approach.

1) I have used Tfidfvectorizer for generating the feature vectors of ingredients because since we are using just the constituents but don't actualy need to derive any semantic relation between the ingredients of other recipe files.
If it was a sentence with some sematic meaning then I would have used BERT or word2vec embeddings.

2)After comparing accuracies of several models on the json data. I have choosen RandomForestClassifier as my classifier because it was giving me better accuracy than other models on this dataset.

3) I am not using KNN classifier so for getting the closest possible dishes , I am using cosine_similarity technique  for getting the similar dishes. N value over here can be anything.

4) For testing I have randomly choosen input and predicted the cuisine type for it using the above described method.
Then for displaying the closest dishes possible for the provided input , I am using cosine_similarity and displaying the top N dishes, that are possibly similar.
