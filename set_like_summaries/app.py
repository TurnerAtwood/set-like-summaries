from flask import Flask, redirect, render_template, request, url_for
from project import get_articles, summarize

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
  if request.method == 'POST':
    article_type = request.form['article_type']
    return redirect(url_for(article_type))
  elif request.method == 'GET':
    return render_template('home.html')


@app.route('/library')
def library():
  return render_template('library.html')


@app.route('/articles', methods=['GET', 'POST'])
def articles():
  if request.method == 'POST':
    topic_index = request.form['topic_selection']
    article1_bias = request.form['article1_selection']
    article2_bias = request.form['article2_selection']
    a1, a2, their_summary = get_articles(topic_index, article1_bias, article2_bias)
  else:
    a1 = a2 = "Paste Custom Article Here."
    their_summary = "Paste Your Summary Here for Comparison"

  return render_template('articles.html', article1=a1, article2=a2, summary=their_summary)


@app.route('/summary', methods=['GET', 'POST'])
def summary():
  article1 = request.form['article1']
  article2 = request.form['article2']
  operation_choice = request.form['operation']
  num_sentences = int(request.form['number_sentences'])
  their_summary = request.form['their_summary']

  our_summary, scores = summarize(article1, article2, operation_choice, num_sentences, their_summary)

  return render_template('summary.html', our_summary=our_summary, their_summary=their_summary, scores=scores)


if __name__ == '__main__':
  app.run(debug=True)