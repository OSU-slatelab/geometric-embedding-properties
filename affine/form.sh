#srun -w osmium -c 20 --mem 30000 -J affine python3 transform.py ../geo-emb/pretrained-embeddings/top_10000_emb.txt Word2Vec
#srun -w osmium -c 20 --mem 30000 -J affine python3 transform.py ../geo-emb/pretrained-embeddings/GoogleNews-vectors-negative300.bin Word2Vec
#srun -w osmium -c 20 --mem 30000 -J affine python3 transform.py ../geo-emb/pretrained-embeddings/glove.840B.300d.word2vec_clean.bin Word2Vec
#srun -w osmium -c 20 --mem 30000 -J affine python3 transform.py ../geo-emb/pretrained-embeddings/wiki-news-300d-1M-subword.bin Word2Vec
#srun -w osmium -c 20 --mem 30000 -J affine python3 transform.py ../geo-emb/pretrained-embeddings/first-100__source--GoogleNews-vectors-negative300.txt Word2Vec
#srun -w osmium -c 20 --mem 30000 -J linearB python3 rand_linear.py ../geo-emb/pretrained-embeddings/parse-error-fix_glove.840B.300d.word2vec_clean.bin Word2Vec
#srun -w osmium -c 20 --mem 30000 -J linearB python3 rand_linear.py ../geo-emb/pretrained-embeddings/GoogleNews-vectors-negative300.bin Word2Vec
srun -w osmium -c 20 --mem 30000 -J linearB python3 rand_linear.py ../geo-emb/pretrained-embeddings/wiki-news-300d-1M-subword.bin Word2Vec
