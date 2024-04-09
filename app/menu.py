from chromaDB import (
    get_collection,
    get_all_tweets,
    get_n_tweets,
    print_collection,
    save_result,
)

def get_count_collection(collection):
    print("###########################\n")
    print("0. Get count collection\n")
    print("1. Dont get\n")
    print("###########################\n")

    option = int(input())
    match option:
        case 0:
            print("Count collection: ", collection.count())
        case _:
            return
        
def get_list_collection(client):
    print("###########################\n")
    print("0. Get list collection\n")
    print("1. Dont get\n")
    print("###########################\n")

    option = int(input())
    match option:
        case 0:
            print("Count collection: ", client.list_collections())
        case _:
            return

def choose_tweets(collection, search_text, hf_token):
    print("###########################\n")
    print("0. Get all tweets from collection\n")
    print("1. Get n tweets from collection\n")
    print("###########################\n")

    option = int(input())
    match option:
        case 0:
            return get_all_tweets(collection, search_text, hf_token), collection.count()
        case 1:
            n_results = int(input("Write a number of tweets to search \n"))
            return get_n_tweets(collection, search_text, n_results, hf_token), n_results
        case _:
            return None
        
def show_tweets(result, n_results):
    print("###########################\n")
    print("0. Print in terminal\n")
    print("1. Save in txt file: ./results\n")
    print("###########################\n")
    option = int(input())

    match option:
        case 0:
            print_collection(result, n_results)
        case 1:
            save_result(result, n_results)
        case _:
            return None

def show_menu():
    hf_token = str(input("Write a token from (https://huggingface.co/join) \n"))
    chroma_db_path = str(input("Write a chromaDB path \n"))
    collection_name = str(input("Write a collection name (https://huggingface.co/join) \n"))
    collection, client = get_collection(chroma_db_path, collection_name)
    search_text = str(input("Write a text to search \n"))

    result, n_results = choose_tweets(collection, search_text, hf_token)
    if result is not None:
        show_tweets(result, n_results)
        
    get_count_collection(collection)
    get_list_collection(client)