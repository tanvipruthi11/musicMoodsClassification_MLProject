## Final Report Documentation
We have submitted the detailed project report for our project work. This includes all the different areas that the team explored and have gathered while working on the project. We have also tried to relate the concepts being thought from the class into our project and show different variations in our production model architecture and the details on how we concluded the architecture of our ANN model.

## Poster
We have briefly highlighted the key findings and insights in a poster that summarizes our work.

The below details consists of all the files that we have included in our submission.

## dataset folder 
1) dataser_1.csv : This dataset contains audio features across two labels - happy and sad. It has around 32k+ records
2) spotify_output_dataset.csv : This dataset is manually prepared by extracting the audio songs from around 50 playlist that is prepared manually. This dataset has around 3000 records.

## weeklysubmissions folder
1) Project Proposal document
2) Updated Project proposal.
3) playlist_spotify_data.py : Python script to extract audio features from the prepared playlists url's.
4) spotify-playlist.xlsx: This has the list of playlist nad the urls to those playlist along with the number of songs extracted from the respective playlist.
5) spotify_connect.py : The code on how to connect to spotify developer's api.
6) sample_cnn.py : Initial CNN model to explore and test a model on our dataset.
7) Synthethic_Data_Generation.ipynb: We explored GAN model to prepare and generate synthethic dataset based on our audio features data. This file contains our research work on the same. We did not use it in our final model to reduce the complexity.
8) Part1_Exploratory_analysis_on_dataset.ipynb: This file contains our initial model data preparation and exploratory analysis on different features present in audio songs. We did some findings on how the features are correlated and how they play role in classifying the moods of the songs.
9) Part2_Group12_Project_Submission.ipynb: This file includes our implementation work on some classical ML models and how our dataset behaved on these models. We also explored SMOTE analysis here.
10) Part3_Part2_continuation_Group12_Project_Submission.ipynb: This file includes our work on top of Part2 submission, where we improved the parameters of RFC,SVM, and implemented PCA analysis.
11) Part4_ANNModel_Group12_Project_Submission.ipynb: This file includes the implementation of ANN model along with multiple variations in the architecture and how we concluded with our final production model.

## src folder
1) ANN_Eightlabels.ipynb : This contains our production model code and results when executed on the spotify_output_dataset.csv file, which contains eight labels data.
2) ANN_TwoLabels.ipynb : This shows our production model code when executed with kaggle dataset having two classes (happy and sad), with around 32k+ records.
3) ClassicalModels_EightLabels.ipynb : This shows the performance of the classical models (KNN, RFC, SVM) behaviour on the dataset having eight labels. 
4) ClassicalModels_TwoLabels.ipynb : This shows the performance of the classical models (KNN, RFC, SVM) behaviour on the dataset having twp labels.
5) incorrect_songs_analysis.xlsx: This contains the analysis of the misclassified songs. The analysis was done by listening to the songs and the features and lyrics and finding insights on the reason why the model would have misclassified it.
