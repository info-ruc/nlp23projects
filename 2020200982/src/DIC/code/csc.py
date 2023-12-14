import sys
import pinyin
import jieba
import string
import re

FILE_PATH = "../dataset/frequent_position.txt"
PUNCTUATION_LIST = string.punctuation
PUNCTUATION_LIST += "。，？：；｛｝［］‘“”《》／！％……（）"


def construct_dict( file_path ):
	
	word_freq = {}
	with open(file_path, "r", encoding='utf-8') as f:
		for line in f:
			info = line.split()
			word = info[0]
			frequency = info[1]
			word_freq[word] = frequency
	
	return word_freq

def construct_examples(file):
	examples = [[], []]
	with open(file, "r", encoding='utf-8') as f:
		for line in f:
			temp = line.strip().split('\t')
			examples[0].append(temp[0])
			examples[1].append(temp[1])
	return examples

def load_cn_words_dict( file_path ):
	cn_words_dict = ""
	with open(file_path, "r", encoding='utf-8') as f:
		for word in f:
			cn_words_dict += word.strip()
	return cn_words_dict


def edits1(phrase, cn_words_dict):
	"All edits that are one edit away from `phrase`."
	phrase = phrase
	splits     = [(phrase[:i], phrase[i:])  for i in range(len(phrase) + 1)]
	deletes    = [L + R[1:]                 for L, R in splits if R]
	transposes = [L + R[1] + R[0] + R[2:]   for L, R in splits if len(R)>1]
	replaces   = [L + c + R[1:]             for L, R in splits if R for c in cn_words_dict]
	inserts    = [L + c + R                 for L, R in splits for c in cn_words_dict]
	return set(deletes + transposes + replaces + inserts)

def known(phrases): return set(phrase for phrase in phrases if phrase in phrase_freq)


def get_candidates( error_phrase ):
	
	candidates_1st_order = []
	candidates_2nd_order = []
	candidates_3nd_order = []
	
	error_pinyin = pinyin.get(error_phrase, format="strip", delimiter="/")
	cn_words_dict = load_cn_words_dict( "../dataset/cn_dic.txt" )
	candidate_phrases = list( known(edits1(error_phrase, cn_words_dict)) )
	
	for candidate_phrase in candidate_phrases:
		candidate_pinyin = pinyin.get(candidate_phrase, format="strip", delimiter="/")
		if candidate_pinyin == error_pinyin:
			candidates_1st_order.append(candidate_phrase)
		elif candidate_pinyin.split("/")[0] == error_pinyin.split("/")[0]:
			candidates_2nd_order.append(candidate_phrase)
		else:
			candidates_3nd_order.append(candidate_phrase)
	
	return candidates_1st_order, candidates_2nd_order, candidates_3nd_order


def auto_correct( error_phrase ):
	
	c1_order, c2_order, c3_order = get_candidates(error_phrase)
	if c1_order:
		return max(c1_order, key=phrase_freq.get )
	elif c2_order:
		return max(c2_order, key=phrase_freq.get )
	elif c3_order:
		return max(c3_order, key=phrase_freq.get )
	else:
		return error_phrase

def auto_correct_sentence( error_sentence, verbose=True):
	
	jieba_cut = jieba.cut( error_sentence, cut_all=False)
	seg_list = "\t".join(jieba_cut).split("\t")
	
	correct_sentence = ""
	i = 0
	while i < len(seg_list):
		phrase = seg_list[i]
		correct_phrase = phrase
		# check if item is a punctuation
		if phrase not in PUNCTUATION_LIST:
			# check if the phrase in our dict, if not then it is a misspelled phrase
			if phrase not in phrase_freq.keys():
				correct_phrase = auto_correct(phrase)
				if verbose :
					print(phrase, correct_phrase)
				correct_sentence += correct_phrase
			else:
				# Non-Optimized version
				# correct_sentence += phrase
				# i += 1
				# continue

				# Optimized version
				# in the dict, but still could be a mistake
				if len(phrase) == 1:
					comb = []
					if i > 0:
						comb.append(seg_list[i-1] + phrase)
					if i < len(seg_list) - 1:
						comb.append(phrase + seg_list[i+1])

					flag = 0

					for c in comb:
						if flag == 1:
							break
						for e in range(len(phrase_pron_examples[0])):
							if flag == 1:
								break
							if c in phrase_pron_examples[0][e]:
								# Find one wrong use example
								index = phrase_pron_examples[0][e].find(c)
								correct_phrase = phrase_pron_examples[1][e][index:index+len(c)]
								flag = 1
								# if c == xxx + phrase
								if c.find(phrase) != 0:
									correct_sentence = correct_sentence[:-c.find(phrase)]
									correct_sentence += correct_phrase
								else:
									correct_sentence += correct_phrase
									char_left = len(c) - len(phrase)
									i += 1
									while char_left > 0:
										if seg_list[i] != '':
											seg_list[i] = seg_list[i][:-1]
											char_left -= 1
										if seg_list[i] == '':
											i += 1
									i -= 1
					if flag == 0:
						correct_sentence += correct_phrase
				else:
						correct_sentence += correct_phrase

					# if flag == 0:
					# 	correct_sentence += correct_phrase
					# else:

		else:
			# if is punctuation, then add it to the correct sentence
			correct_sentence += correct_phrase

		i += 1

	return correct_sentence


phrase_freq = construct_dict( FILE_PATH )
phrase_pron_examples = construct_examples('../dataset/train.txt')

def main():
	while True:
		err_sent = input("Please input a sentence:\n")
		correct_sent = auto_correct_sentence( err_sent )
		print("WordCut:\t" + ",".join(jieba.cut(err_sent, cut_all=False)))
		print("original sentence:" + err_sent + "\n==>\n" + "corrected sentence:" + correct_sent)
	
if __name__=="__main__":
	main()
