#! /usr/bin/python3

import sys, os
import re
from xml.dom.minidom import parse
import spacy

import paths
from dictionaries import Dictionaries

## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML
def get_label(tks, tke, spans) :
    for (spanS,spanE,spanT) in spans :
        if tks==spanS and tke<=spanE+1 : return "B-"+spanT
        elif tks>spanS and tke<=spanE+1 : return "I-"+spanT
    return "O"
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_sentence_features(tokens, dicts) :

   def word_shape(txt) :
      shape = []
      for ch in txt :
         if ch.isupper() : shape.append('A')
         elif ch.islower() : shape.append('a')
         elif ch.isdigit() : shape.append('d')
         else : shape.append(ch)
      return "".join(shape)

   # for each token, generate list of features and add it to the result
   sentenceFeatures = {}
   for i,tk in enumerate(tokens) :
      tokenFeatures = []
      t = tk.text

      tokenFeatures.append("form="+t)
      tokenFeatures.append("formlower="+t.lower())
      tokenFeatures.append("pref2="+t[:2])
      tokenFeatures.append("pref3="+t[:3])
      tokenFeatures.append("pref4="+t[:4])
      tokenFeatures.append("suf3="+t[-3:])
      tokenFeatures.append("suf4="+t[-4:])
      tokenFeatures.append("shape="+word_shape(t))
      tl = len(t)
      if tl <= 3 : tokenFeatures.append("len<=3")
      elif tl <= 6 : tokenFeatures.append("len<=6")
      elif tl <= 10 : tokenFeatures.append("len<=10")
      else : tokenFeatures.append("len>10")
      if len(t) >= 2 :
         tokenFeatures.append("ch2="+t[:2])
         tokenFeatures.append("ch2="+t[-2:])
      if len(t) >= 3 :
         tokenFeatures.append("ch3="+t[:3])
         tokenFeatures.append("ch3="+t[-3:])
      if t.isupper() : tokenFeatures.append("isUpper")
      if t.istitle() : tokenFeatures.append("isTitle")
      if t.isdigit() : tokenFeatures.append("isDigit")
      if '-' in t : tokenFeatures.append("hasDash")
      if re.search('[0-9]',t) : tokenFeatures.append("hasDigit")
      if '(' in t or ')' in t : tokenFeatures.append("hasParen")
      if '/' in t : tokenFeatures.append("hasSlash")
      if '.' in t : tokenFeatures.append("hasDot")
      if '+' in t : tokenFeatures.append("hasPlus")
      if ',' in t : tokenFeatures.append("hasComma")
      found,val = dicts.find(t.lower(), 'external')
      if found:
         tokenFeatures.append("inDictFull")
         for c in val : tokenFeatures.append("external="+c)
      found,val = dicts.find(t.lower(), 'externalpart')
      if found:
          tokenFeatures.append("inDictPart")
          for c in val : tokenFeatures.append("externalpart="+c)

      if i>0 :
         tPrev = tokens[i-1].text
         tokenFeatures.append("formPrev="+tPrev)
         tokenFeatures.append("formlowerPrev="+tPrev.lower())
         tokenFeatures.append("pref2Prev="+tPrev[:2])
         tokenFeatures.append("pref3Prev="+tPrev[:3])
         tokenFeatures.append("pref4Prev="+tPrev[:4])
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
         tokenFeatures.append("suf4Prev="+tPrev[-4:])
         tokenFeatures.append("shapePrev="+word_shape(tPrev))
         if tPrev.isupper() : tokenFeatures.append("isUpperPrev")
         if tPrev.istitle() : tokenFeatures.append("isTitlePrev")
         if tPrev.isdigit() : tokenFeatures.append("isDigitPrev")
         if '-' in tPrev : tokenFeatures.append("hasDashPrev")
         if re.search('[0-9]',tPrev) : tokenFeatures.append("hasDigitPrev")
         if '(' in tPrev or ')' in tPrev : tokenFeatures.append("hasParenPrev")
         if '/' in tPrev : tokenFeatures.append("hasSlashPrev")
         if '.' in tPrev : tokenFeatures.append("hasDotPrev")
         if '+' in tPrev : tokenFeatures.append("hasPlusPrev")
         if ',' in tPrev : tokenFeatures.append("hasCommaPrev")
         found,val = dicts.find(tPrev.lower(), 'external')
         if found:
             tokenFeatures.append("inDictFullPrev")
             for c in val : tokenFeatures.append("externalPrev="+c)
         found,val = dicts.find(tPrev.lower(), 'externalpart')
         if found:
             tokenFeatures.append("inDictPartPrev")
             for c in val : tokenFeatures.append("externalpartPrev="+c)
      else :
         tokenFeatures.append("BoS")

      if i<len(tokens)-1 :
         tNext = tokens[i+1].text
         tokenFeatures.append("formNext="+tNext)
         tokenFeatures.append("formlowerNext="+tNext.lower())
         tokenFeatures.append("pref2Next="+tNext[:2])
         tokenFeatures.append("pref3Next="+tNext[:3])
         tokenFeatures.append("pref4Next="+tNext[:4])
         tokenFeatures.append("suf3Next="+tNext[-3:])
         tokenFeatures.append("suf4Next="+tNext[-4:])
         tokenFeatures.append("shapeNext="+word_shape(tNext))
         if tNext.isupper() : tokenFeatures.append("isUpperNext")
         if tNext.istitle() : tokenFeatures.append("isTitleNext")
         if tNext.isdigit() : tokenFeatures.append("isDigitNext")
         if '-' in tNext : tokenFeatures.append("hasDashNext")
         if re.search('[0-9]',tNext) : tokenFeatures.append("hasDigitNext")
         if '(' in tNext or ')' in tNext : tokenFeatures.append("hasParenNext")
         if '/' in tNext : tokenFeatures.append("hasSlashNext")
         if '.' in tNext : tokenFeatures.append("hasDotNext")
         if '+' in tNext : tokenFeatures.append("hasPlusNext")
         if ',' in tNext : tokenFeatures.append("hasCommaNext")
         found,val = dicts.find(tNext.lower(), 'external')
         if found:
            tokenFeatures.append("inDictFullNext")
            for c in val : tokenFeatures.append("externalNext="+c)
         found,val = dicts.find(tNext.lower(), 'externalpart')
         if found:
            tokenFeatures.append("inDictPartNext")
            for c in val : tokenFeatures.append("externalpartNext="+c)
      else:
         tokenFeatures.append("EoS")

      if i == 0 : tokenFeatures.append("isFirst")
      if i == 1 : tokenFeatures.append("isSecond")
      if i == len(tokens)-2 : tokenFeatures.append("isSecondLast")
      if i == len(tokens)-1 : tokenFeatures.append("isLast")
    
      sentenceFeatures[i] = tokenFeatures
    
   return sentenceFeatures

## --------- Feature extractor ----------- 
## -- Extract features for each token in each
## -- sentence in each file of given dir

def extract_features(datafile, outfile) :

    # load dictionaries
    dicts = Dictionaries(os.path.join(paths.RESOURCES,"dictionaries.json"))

    # open output file
    outf = open(outfile, "w")
    
    # create analyzer. We don't need the parser now, it will be faster if disabled
    nlp = spacy.load("en_core_web_trf", enable=["tokenizer"])
    
    # parse XML file, obtaining a DOM tree
    tree = parse(datafile)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      print(f"extracting sentence {sid}        \r", end="")
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity") # get gold standard entities
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))

      # convert the sentence to a list of tokens
      tokens = nlp(stext)
      # extract sentence features
      features = extract_sentence_features(tokens, dicts)

      # print features in format expected by CRF/SVM/MEM trainers
      for i,tk in enumerate(tokens) :
         # see if the token is part of an entity
         tks,tke = tk.idx, tk.idx+len(tk.text)
         # get gold standard tag for this token
         tag = get_label(tks, tke, spans)
         # print feature vector for this token
         print (sid, tk.text, tks, tke-1, tag, "\t".join(features[i]), sep='\t', file=outf)

      # blank line to separate sentences
      print(file=outf)

    # close output file
    outf.close()

## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir outfile
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- corresponding feature vectors to outfile
## --

if __name__ == "__main__" :
    # directory with files to process
    datafile = sys.argv[1]
    # file where to store results
    featfile = sys.argv[2]
    
    extract_features(datafile, featfile)

