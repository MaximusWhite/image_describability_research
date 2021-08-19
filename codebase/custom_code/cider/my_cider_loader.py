from my_cider import CiderChad



cider_scorer = CiderChad(df_config=[None, 1])

print(cider_scorer.test('a nasty chair sitting in the middle of the room'))