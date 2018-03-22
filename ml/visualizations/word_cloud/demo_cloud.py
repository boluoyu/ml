import matplotlib.pyplot as plt

from argparse import ArgumentParser
from wordcloud import WordCloud, STOPWORDS


def main(text_file_path):
    print('hey')
    text = open(text_file_path).read()

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=STOPWORDS).generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40, stopwords=STOPWORDS).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--text_file_path', required=True)

    args = parser.parse_args()
    main(args.text_file_path)
