# self-learning-llm

Code adapted from https://github.com/teddy-f-47/self-learning-llm-public

Main code in `uncertainty`, scripts to run are in `scripts`

**Requirements**
Additional packages from preivous code include `openai`.

**Layout**

`main.py` does the following

1) Generate topic embedding space and samples `num_iterations`*`topic_ratio` where topic ratio = questions/topic. Saved
2) Generate `num_iterations` questions as well as greedy and sampled answers. Questions are scored by choice of `scoring_method` and filtered with top `filter_size` are used to form Q_H
3) Generate chosen labels for DPO using the choice of `answer_generator`.
4) Perform training by setting `training` to true.
5) Perform testing by setting `testing` to true. The `test_keys`:'answer_confidence' evaluates confidence on test set, 'question_confidence' on confidence score of generated questions using same topic as training set. `answer_performance` uses GPT4 to score pairwise ranking.

**Faster Inference**

The code uses TGI for step 2 and 3(for self-generated), which is much faster than standard HF generate function.

