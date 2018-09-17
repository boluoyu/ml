import kenlm
model = kenlm.LanguageModel('/Users/jnewman/Projects/learning/ai_blog/bible.klm')
print(model.score('in the beginning was the word'))