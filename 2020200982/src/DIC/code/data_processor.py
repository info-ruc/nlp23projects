from html.parser import HTMLParser
import re
import argparse
import xml.dom.minidom
from xml.dom.minidom import parse
from paddle.utils.download import get_path_from_url

def main():
    get_path_from_url('https://github.com/wdimmy/Automatic-Corpus-Generation/raw/master/corpus/train.sgml',
                      '../dataset/')
    with open("../dataset/train.txt", "w", encoding="utf-8") as fw:
        with open("../dataset/train.sgml", "r", encoding="utf-8") as f:
            input_str = f.read()
            
        # Add fake root node <SENTENCES>
        input_str = "<SENTENCES>" + input_str + "</SENTENCES>"
        dom = xml.dom.minidom.parseString(input_str)
        example_nodes = dom.documentElement.getElementsByTagName("SENTENCE")
        for example in example_nodes:
            raw_text = example.getElementsByTagName("TEXT")[0].childNodes[0].data
            correct_text = list(raw_text)
            mistakes = example.getElementsByTagName("MISTAKE")
            for mistake in mistakes:
                loc = int(mistake.getElementsByTagName("LOCATION")[0].childNodes[0].data) - 1
                correction = mistake.getElementsByTagName("CORRECTION")[0].childNodes[0].data
                correct_text[loc] = correction

            correct_text = "".join(correct_text)
            fw.write("{}\t{}\n".format(raw_text, correct_text))


if __name__ == "__main__":
    main()
