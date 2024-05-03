import nltk
nltk.download('treebank')

print("Number of sentences in the Treebank: ", len(nltk.corpus.treebank.parsed_sents()))
print("Generating CFG rules...")

ruleset = set(rule for tree in nltk.corpus.treebank.parsed_sents()
           for rule in tree.productions())

print("Number of CFG rules: ", len(ruleset))
print("Writing CFG rules to preprocess/cfg.txt...")

with open("preprocess/cfg.txt", "w") as f:
    for rule in ruleset:
        f.write(str(rule) + "\n")
        
print("Done!")