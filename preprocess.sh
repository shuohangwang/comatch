task="race"
RACE="data/race/RACE.tar.gz"
GloVe="data/glove/glove.840B.300d.txt"

if [ ! -f "$GloVe" ]; then
	wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P data/glove
	unzip -o -d data/glove/ data/glove/glove.840B.300d.zip
fi;

if [ ! -f "trainedmodel" ]; then
    mkdir trainedmodel
fi;

if [ "$task" = "race" ]; then
    if [ ! -d "data" ]; then
        mkdir data
    fi;
    if [ ! -d "data/race" ]; then
        mkdir data/race/
    fi;
    if [ ! -d "data/wikiqa/sequence" ]; then
        mkdir data/race/sequence
    fi;

    if [ -f "$RACE" ]; then
        tar -xf $RACE -C data/race/
        python preprocess.py
    else
        echo "!!!!!!!!Please dowload the file \"RACE.tar.gz\" to the path data/race/ through address: https://www.cs.cmu.edu/~glai1/data/race/"
    fi;
fi;
