cd ../src/main/python
python predict.py --data=../../../data/corpus/eval/earnings22.txt\
                  --config=../../../config/config.json\
                  --model=../../../models/model.torch\
                  --vocabulary=../../../models/vocabulary.pkl\
                  --output=../../../output/corpus-probabilities.tsv
