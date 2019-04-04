FILE_LIST=$(ls movie_script_txts/)
for script in $FILE_LIST; do
    NUM=$(ls movie_script_txts | wc -l)
    echo $NUM
    if [ $NUM -gt 269 ]
    then
        mv "movie_script_txts/$script" script_corpus_eval
    fi
done
