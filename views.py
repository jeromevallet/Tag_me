from flask import Flask, render_template, flash, redirect
import pandas as pd
import numpy as np
import scipy
import datetime
from datetime import timedelta, date
import pickle

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.widgets import TextArea
from wtforms.validators import DataRequired

# Project 4
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Project 6
import re
import nltk
nltk.download('tokenizers/punkt/PY3/english.pickle')
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Config options - Make sure you created a 'config.py' file.
app.config.from_object('config')

# Project 4
cie = app.config['CIE']
origin_dest = app.config['ORIGIN_DEST']
o_nbflights = app.config['O_NBFLIGHTS']
d_nbflights = app.config['D_NBFLIGHTS']
flnum = app.config['FLNUM']
mois_lib = app.config['MOIS_LIB']

# Project 6
X_train = app.config['X_TRAIN']
X_train['Tags_final'] = X_train['Tags_final'].apply(lambda x: x.split())

# Project 4
SGDR_ridge = pickle.load(open('/home/jvallet/app/models/ridge_flights_7pc.sav', 'rb'))
STD_SCALE = pickle.load(open('/home/jvallet/app/models/scaler_flights_pc7.sav', 'rb'))

# Project 6
main_tags = pickle.load(open('/home/jvallet/app/modelsP6/main_tags.pkl', 'rb'))
count_vect_code = pickle.load(open('/home/jvallet/app/modelsP6/count_vect_code.pkl', 'rb'))
lda_code = pickle.load(open('/home/jvallet/app/modelsP6/lda_code.pkl', 'rb'))
X_train_lda_code = pickle.load(open('/home/jvallet/app/modelsP6/X_train_lda_code.pkl', 'rb'))
count_vect_text = pickle.load(open('/home/jvallet/app/modelsP6/count_vect_text.pkl', 'rb'))
lda_text = pickle.load(open('/home/jvallet/app/modelsP6/lda_text.pkl', 'rb'))
X_train_lda_text = pickle.load(open('/home/jvallet/app/modelsP6/X_train_lda_text.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('/home/jvallet/app/modelsP6/tfidf_vectorizer.pkl', 'rb'))
mlb = pickle.load(open('/home/jvallet/app/modelsP6/mlb.pkl', 'rb'))
classifier = pickle.load(open('/home/jvallet/app/modelsP6/LRMulti.pkl', 'rb'))

class LoginForm(FlaskForm):
    select = []
    for option in range(1,32):
        select.append((str(option), str(option)))
    date_j = SelectField(choices=select)
    select = []
    for option in range(1,13):
        select.append((str(option), str(option)))
    date_m = SelectField(choices=select)
    select = []
    for option in range(0,24):
        select.append((str(option), str(option)))
    heure = SelectField(choices=select)
    select = []
    for option in range(0,60):
        select.append((str(option), str(option)))
    minutes = SelectField(choices=select)
    select = []
    for option in sorted(origin_dest['ORIGIN'].unique()):
        select.append((str(option), str(option)))
    o_aeroport = SelectField(choices=select)
    select = []
    for option in sorted(origin_dest['DEST'].unique()):
        select.append((str(option), str(option)))
    d_aeroport = SelectField(choices=select)
    select = []
    for option in cie:
        select.append((option, option))
    carrier = SelectField(choices=select)
    flnum = StringField(validators=[DataRequired()])
    submit = SubmitField('Evaluer le retard')

class TagsForm(FlaskForm):
    title = StringField(validators=[DataRequired()])
    body = StringField(widget=TextArea())
    submit = SubmitField('Suggérer des tags')

@app.route('/')
def index():
    return "Merci d'indiquer l'application désirée (exemple /delay_app/)"

@app.route('/tags/', methods=['GET', 'POST'])
def appliP6():
    form = TagsForm()

    if form.validate_on_submit():
        title = form.title.data
        body = form.body.data

        df = pd.DataFrame()
        df = df.append({'Title': title, 'Body': body}, ignore_index=True)

        df['Body_length'] = df['Body'].str.len()
        df['Title_length'] = df['Title'].str.len()
        df['Text_code'] = df['Body'].apply(lambda x: re.findall('<pre>.*?</pre>',  x.replace('\\n', ' ').replace('\\t', ' ').replace("\\'", ' ')))
        df['Text_code'] = df['Text_code'].apply(lambda x: [l.replace("<pre><code>", '').replace("</code></pre>", '') for l in x])
        df['Nb_code_block'] = df['Text_code'].apply(lambda x: len(x))
        df['Text'] = df['Title'] + '. ' + df['Body']

        def clean_code_link_html(raw):
            raw = raw.replace('\\n', ' ')
            raw = raw.replace('\\t', ' ')
            raw = raw.replace("\\'", ' ')
            raw = re.sub('<pre>.*?</pre>', ' ', raw, flags=re.MULTILINE)
            raw = re.sub('<a[^>]+>.*?</a>', ' ', raw)
            raw = re.sub('\S*@\S*\s?', ' ', raw)
            raw = raw.lower()
            return BeautifulSoup(raw, 'html.parser').get_text()

        df['Text_cleaned'] = df['Text'].apply(clean_code_link_html)
        df['Text_cleaned'] = df['Text_cleaned'].apply(lambda x: x.replace('#', 'diiiese'))
        df['Text_words'] = df.Text_cleaned.apply(nltk.word_tokenize)
        df['Text_words'] = df['Text_words'].apply(lambda x: [word.replace('diiiese', '#') for word in x])
        # restricting words to alpha + nonalpha_to_use words
        df['Text_words'] = df['Text_words'].apply(lambda x: [word for word in x if re.search(r'^[a-z.+#-]', word)])
        df['Text_words'] = df['Text_words'].apply(lambda x: [word for word in x
                                                                         if ((word!='.') & (word!='+') & (word!='#') & (word!='-'))])
        # removing stopwords
        stopwords = nltk.corpus.stopwords.words('english')
        df['Text_words'] = df['Text_words'].apply(lambda x: [word for word in x if word not in stopwords])
        df['Text_maintags'] = df['Text_words'].apply(lambda x: [word for word in x if word in main_tags])
        df['Text_words'] = df['Text_words'].apply(lambda x: [word for word in x if word not in main_tags])
        # lemmatization
        wnl = nltk.WordNetLemmatizer()
        df['Text_words_lemma'] = df['Text_words'].apply(lambda x: [wnl.lemmatize(word) for word in x])

        def all_postags(tokens):
            all_postags = [(token,pos) for token, pos in nltk.pos_tag(tokens)]
            return all_postags

        df['Text_postags'] = df['Text_words_lemma'].apply(all_postags)

        # selecting just nouns as aspiring keywords
        def just_nouns(tokens):
            nouns = [token for token, pos in nltk.pos_tag(tokens) if pos == 'NN']
            return nouns

        df['Text_final_words'] = df['Text_words_lemma'].apply(just_nouns)
        # and we add the words preserved from lemmatization
        df['Text_final_words'] = df['Text_final_words'] + df['Text_maintags']
        df['Text_code_final_words'] = df.Text_code.apply(lambda x: ' '.join(x)).apply(nltk.word_tokenize)
        df['Text_count'] = df['Text_final_words'].apply(lambda x: len(set(x)))

        def lexical_diversity(text):
            if len(text) > 0:
                return len(set(text)) / len(text)
            else:
                return 0

        # Code part treatment
        df['Text_ld'] = df['Text_words'].apply(lexical_diversity)
        df['Text_final_words_str'] = df['Text_final_words'].apply(lambda x: ' '.join(x))
        df['Text_code_final_words_str'] = df['Text_code_final_words'].apply(lambda x: ' '.join(x))

        df_counts = count_vect_code.transform(df['Text_code_final_words_str'])
        df_lda = lda_code.transform(df_counts)
        X_dist = pairwise_distances(X_train_lda_code, df_lda, metric='cosine')

        def extract_tags_from_nearest_posts(col, post, nb_tags=10):
            tags_extracted = []
            if len(df.loc[post, 'Text_code_final_words']) > 0:
                for i in col.argsort()[:50]:
                    if len(X_train.loc[i, 'Tags_final']) > 0:
                        if (len(tags_extracted) < nb_tags):
                            if X_train.loc[i, 'Tags_final'][0] not in tags_extracted:
                                tags_extracted.append(X_train.loc[i, 'Tags_final'][0])
            return tags_extracted

        # tags_array = []
        # for test_post in range(len(df)):
        #     tags_array.append(extract_tags_from_nearest_posts(X_dist[:, test_post], test_post))
        LDA_tags_from_code = extract_tags_from_nearest_posts(X_dist[:, 0], 0)
        # df['LDA_tags_from_code'] = tags_array

        # Text part treatment
        df_counts = count_vect_text.transform(df['Text_final_words_str'])
        df_lda = lda_text.transform(df_counts)
        X_dist = pairwise_distances(X_train_lda_text, df_lda, metric='cosine')

        def extract_tags_from_nearest_posts(col, post, nb_tags=10):
            tags_extracted = []
            if len(df.loc[post, 'Text_final_words']) > 0:
                for i in col.argsort()[:50]:
                    if len(X_train.loc[i, 'Tags_final']) > 0:
                        if (len(tags_extracted) < nb_tags):
                            if X_train.loc[i, 'Tags_final'][0] not in tags_extracted:
                                tags_extracted.append(X_train.loc[i, 'Tags_final'][0])
            return tags_extracted

        # tags_array = []
        # for test_post in range(len(df)):
        #     tags_array.append(extract_tags_from_nearest_posts(X_dist[:, test_post], test_post))
        LDA_tags_from_text = extract_tags_from_nearest_posts(X_dist[:, 0], 0)

        # df['LDA_tags_from_text'] = tags_array

        # Aggregating Code & Text Tags
        def select_mode_tags(row):
            w_count = {}
            for word in row:
                if word in w_count:
                    w_count[word] += 1
                else:
                    w_count[word] = 1
            mode_words = sorted(w_count, key=w_count.get, reverse=True)
            mode_words_str = ' '.join(mode_words[:10])
            return mode_words_str

        lda_tags = select_mode_tags(LDA_tags_from_code + LDA_tags_from_text)
        lda_tags = lda_tags.split()
        lda_tags = '  |  '.join(lda_tags)

        flash('>> TAGS ISSUS DE LA MODELISATION NON SUPERVISEE LDA <<', '20')
        flash(lda_tags, '20')
        flash('____________________________________________________', '16')

        # Multi-label LR
        df_tfidf = tfidf_vectorizer.transform(df['Text_final_words_str'] + ' ' + df['Text_code_final_words_str'])

        final_test = scipy.sparse.hstack([
                df_tfidf,
                df[['Body_length','Title_length','Nb_code_block','Text_count','Text_ld']]
        ], format='csr')

        y_pred = classifier.predict_proba(final_test)
        pp_df = pd.DataFrame(y_pred, columns=mlb.classes_)

        def extract_tags_from_multiclassifier(row):
            tags_extracted = []
            for tup in sorted(row.to_dict().items(), key=operator.itemgetter(1), reverse=True):
                if len(tags_extracted) < 10:
                    tags_extracted.append(tup[0])
            return tags_extracted

        import operator
        lr_tags = pp_df.apply(extract_tags_from_multiclassifier, axis=1)
        lr_tags = ' '.join(lr_tags.values[0]).split()
        lr_tags = '  |  '.join(lr_tags)

        flash('>> TAGS ISSUS DE LA MODELISATION SUPERVISEE MULTI-LABELS LR <<', '20')
        flash(lr_tags, '20')
        flash('____________________________________________________', '16')
        flash('Rappel de la question :', '16')
        flash(title, '12')
        flash(body, '12')

        return redirect('/tags/')

    return render_template('tags.html', form=form)

@app.route('/delay_app/', methods=['GET', 'POST'])
def appli():
    form = LoginForm()

    if form.validate_on_submit():
        jour = int(form.date_j.data)
        mois = int(form.date_m.data)
        annee = 2017
        heure = int(form.heure.data)
        minutes = int(form.minutes.data)
        origin = form.o_aeroport.data
        dest = form.d_aeroport.data
        carrier = form.carrier.data
        flightnum = int(form.flnum.data)

        feries_US_2017 = [date(2017,1,1), date(2017,1,18), date(2017,2,15), date(2017,3,25), date(2017,5,30), date(2017,7,4),
                         date(2017,9,5), date(2017,10,10), date(2017,11,11), date(2017,11,24), date(2017,12,25), date(2017,12,26),
                         date(2018,1,1)]

        def nbj_proche_ferie(d):
            x = [(abs(d-f)).days for f in feries_US_2017]
            return min(x)

        df_type = pd.read_csv('/home/jvallet/app/models/DF_TYPE.csv', index_col=0)

        if len(origin_dest.loc[(origin_dest.ORIGIN == origin) & (origin_dest.DEST == dest),'CRS_ELAPSED_TIME'])>0:
            df_type['CRS_ELAPSED_TIME'] = origin_dest.loc[(origin_dest.ORIGIN == origin) & (origin_dest.DEST == dest),'CRS_ELAPSED_TIME'].values[0]
            df_type['DISTANCE'] = origin_dest.loc[(origin_dest.ORIGIN == origin) & (origin_dest.DEST == dest),'DISTANCE'].values[0]
        else:
            flash('Aucun vol recensé entre {} et {} : prédiction impossible'.format(origin, dest))
            return redirect('/delay_app/')

        if (mois == 2) & (jour>28):
            mois = 3
            jour = jour - 28
        elif ((mois == 4) | (mois == 6) | (mois == 9) | (mois == 11)) & (jour == 31):
            mois += 1
            jour = 1
        date_fr = str(jour)+' '+mois_lib[mois]+' 2017'

        df_type['FL_DATE'] = date(annee, mois, jour)
        df_type['GAP_FERIE'] = df_type['FL_DATE'].apply(nbj_proche_ferie).astype(np.int8)
        if len(flnum.loc[(flnum.UNIQUE_CARRIER == carrier) & (flnum.FL_NUM == flightnum),'FL_NUM_DAYS'])>0:
            df_type['FL_NUM_DAYS'] = flnum.loc[(flnum.UNIQUE_CARRIER == carrier) & (flnum.FL_NUM == flightnum),'FL_NUM_DAYS'].values[0]
            df_type['NROTATIONS'] = flnum.loc[(flnum.UNIQUE_CARRIER == carrier) & (flnum.FL_NUM == flightnum),'NROTATIONS'].values[0]
        df_type['ORIGIN_AIRPORT_NBFLIGHTS'] = o_nbflights.loc[o_nbflights.ORIGIN == origin,'ORIGIN_AIRPORT_NBFLIGHTS'].values[0]
        df_type['DEST_AIRPORT_NBFLIGHTS'] = d_nbflights.loc[d_nbflights.DEST == dest,'DEST_AIRPORT_NBFLIGHTS'].values[0]
        df_type['DAY_OF_MONTH'] = jour
        df_type['DAY_OF_WEEK'] = date(annee, mois, jour).weekday()
        df_type['WEEK'] = date(annee, mois, jour).isocalendar()[1]
        df_type['CRS_DEP_HOUR'] = heure
        df_type['CRS_ARR_HOUR'] = round((minutes + df_type['CRS_ELAPSED_TIME'])/60 + heure, 0)
        df_type['CRS_ARR_HOUR'] = df_type['CRS_ARR_HOUR'].apply(lambda x: (x-24) if x>23 else x)
        df_type['CARRIER_AA'] = 0
        df_type['CARRIER_'+carrier] = 1

        cols_utiles = [c for c in df_type.columns if c not in ['FL_DATE']]

        df_type_scaled = STD_SCALE.transform(df_type[cols_utiles])
        retard = round(SGDR_ridge.predict(df_type_scaled)[0],0)

        if retard>=0:
            flash('Retard estimé pour le vol {}{} du {} : {} minutes'.format(form.carrier.data, form.flnum.data, date_fr, retard))
        else:
            flash('Pas de retard prévu pour le vol {}{} du {} (avance estimée {} minutes)'.format(form.carrier.data, form.flnum.data, date_fr, retard))
        return redirect('/delay_app/')

    return render_template('index.html', form=form)

if __name__=="__main__":
    app.run(debug=True)