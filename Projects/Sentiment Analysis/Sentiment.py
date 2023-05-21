from h2o_wave import Q, app, ui, main
import joblib
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

countvector = joblib.load('./countvector.joblib')
model = joblib.load('./sentiment_model.joblib')
lm = joblib.load('./lemmitizer.joblib')

@app('/sentiment_analysis')
async  def serve(q:Q):

    createform(q)

    if q.args.btn_submit:
        analyze(q)

    await q.page.save()


def createform(q:Q):

    # Create header
    q.page['header'] = ui.header_card(
        box='1 1 10 1',
        title='Sentiment Analysis',
        subtitle='Analyze your sentiments here',
        icon='TriangleSolidRight12'
    )

    q.page['form'] = ui.form_card(
        box='1 2 10 2',
        items=[
            ui.textbox(name='txt_sentiment', label='Write down your sentiments', multiline=True),
            ui.button(name='btn_submit', label='Submit', primary=True)
        ]
    )

def text_transformation(df_col):
  corpus = []

  for text in df_col:
    new_text = re.sub('[^a-zA-Z]',' ', str(text))
    new_text = new_text.lower()
    new_text = new_text.split()
    new_text = [lm.lemmatize(word) for word in new_text
                if word not in set(stopwords.words('english'))]
    corpus.append(" ".join(str(word) for word in new_text))

  return corpus

def analyze(q:Q):

    analysis_text = "TextBox cannot be empty :|"

    if q.args.txt_sentiment != "":
        sentence = q.args.text_sentiment
        corpus = text_transformation([sentence])
        transformed = countvector.transform(corpus)
        predict = model.predict(transformed)

        if np.argmax(predict) == 0:
            analysis_text = 'This is a negative statement :-('
        elif np.argmax(predict) == 1:
            analysis_text = ' This is a neutral statement :-|'
        else:
            analysis_text = ' This is a positive statement :-)'

    q.page['analysis']=ui.form_card(
        box='1 4 10 1',
        items=[
            #ui.image(title='Happy', path=image, type='png')
            ui.text(analysis_text, size=ui.TextSize.XL)
        ],

    )