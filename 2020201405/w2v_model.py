from optparse import OptionParser
from gensim.models import word2vec
import logging

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


class W2v(object):
    def __init__(self,fin=None):
        self.model = None
        self.file = fin

    def load(self,fmodel):
        self.model = word2vec.Word2Vec.load(fmodel)

    def process(self,size=100,window=5,min_count=5):
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        workers = cpu_count
        sentences = word2vec.LineSentence(self.file)
        self.model = word2vec.Word2Vec(sentences,vector_size=size,window=window,min_count=min_count,workers=workers)

    def save(self,fout):
        try:
            self.model.save(fout)
        except Exception as e:
            logging(self.fout, "FILE ERROR!")
            exit(0)


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-f', '--file', type=str,  help='语料库文件', dest='wordsfile')
    parser.add_option('-m', '--model', type=str,  help='w2v模型文件', dest='modelfile')
    parser.add_option('-s', '--size', type=int, default=100, help='w2v向量维度',dest='size')

    options, args = parser.parse_args()
    if not (options.wordsfile or options.modelfile):
        parser.print_help()
        exit()

    wordsfile = options.wordsfile
    model = options.modelfile
    size = options.size

    w = W2v(options.wordsfile)
    w.process(size=size)
    w.save(model)