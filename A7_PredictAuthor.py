import os
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

paths = ['Training/' + p for p in os.listdir('Training')]
authors = os.listdir('Training')
# Read files in the Training directory and create a list containing all the books as one book
TrainingData = {authors[i]: [''.join([open(paths[i] + '/' + file, encoding='latin-1').read()[1:]  #skips first line too
                                      for file in os.listdir(paths[i])])] for i in range(len(paths))}
# I don't know what else to filter from the books
stop_words = text.ENGLISH_STOP_WORDS.union({'AUSTEN', 'Jane', 'Austen', 'Frank', 'Baum', 'Jules', 'Verne'})
cvect = text.CountVectorizer(stop_words=stop_words, min_df=3)
train = cvect.fit_transform([TrainingData[auth][0] for auth in authors])
transformer = TfidfTransformer()
trans_train = transformer.fit_transform(train)
naiveBayesClassifier = MultinomialNB().fit(trans_train, authors)

def mainLoop():
    while True:
        print("Enter a file name (including extension) (q or whatever to quit)")
        fileName = input("File: ")
        if fileName in ['q', 'Q', 'Quit', 'quit', 'exit', 'Exit', 'whatever']:
            break
        try:
            selectedFile = [open(fileName, encoding='UTF-8').read()]
            test = cvect.transform(selectedFile)
            trans_test = transformer.transform(test)
            print("Algorithm says the author is:", naiveBayesClassifier.predict(trans_test)[0])
        except FileNotFoundError:
            print("File", fileName, "was not found")
            continue
    exit(0)

mainLoop()