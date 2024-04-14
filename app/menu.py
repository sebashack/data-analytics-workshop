import argparse
from datetime import datetime

from chromaDB import (
    create_client,
    get_or_create_collection,
    get_tweet_by_id,
    count_collection,
    query,
)


def create_parser():
    parser = argparse.ArgumentParser(
                    prog='Tweets Search Engine',
                    description='Operations in ChromaDB model',
    )
    #Operations
    parser.add_argument('-ls', help='list collection information in ChromaDB', action='store_true')
    parser.add_argument('-c', help='get number of tweets in collection', action='store_true')
    parser.add_argument('-sv', help='save tweets in ./output', action='store_true')

    #Variables
    parser.add_argument('-p', help='path in your local machine: --p <chromadb_path> <collection_name> <save_path>', type=str, nargs='+')
    parser.add_argument('-i', help='get tweets by ids (None show all tweets)', type=str, nargs='*')
    parser.add_argument('-t', help='token authentication for https://huggingface.co/join', type=str)
    parser.add_argument('-s', help='text to search', type=str)
    parser.add_argument('-n', help='number of tweets to receive (None return all tweets found)', type=int)

    return parser

def print_collection(result):
    for idx in range(0, len(result)):
        print(f"ID: {result['ids'][0][idx]}, Document: {result['documents'][0][idx]}, Distance: {result['distances'][0][idx]}")
        print("###########################\n")

def save_result(result):
    utc_datetime = datetime.utcnow()
    utc_datetime_str = utc_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    file = open(f"./output__{utc_datetime_str}.txt", "w")
    
    for idx in range(0, len(result)):
        file.write(f"ID: {result['ids'][0][idx]} Document: {result['documents'][0][idx]} Distance: {result['distances'][0][idx]}\n")

    file.close()

def choose_operation(args):
    client = create_client(args.p[0])
    collection = get_or_create_collection(client, args.p[1])
    if args.ls:
        print(collection)
    if args.c:
        print(count_collection(collection))
    if args.i != None:
        print(get_tweet_by_id(args.i, collection))
    if args.s != None and args.t != None:
        n_tweets = args.n
        if n_tweets == None:
            n_tweets = count_collection(collection)

        tweets = query(collection, args.s, args.t, n_tweets)

        if args.sv:
            save_result(tweets)
        else:
            print_collection(tweets)
