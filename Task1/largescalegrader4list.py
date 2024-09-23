import time
from submission import flatten_list

SCALE = 1e2

if __name__ == "__main__":
    
    # produce a list of SCALE elements that need to be flattened
    nested_list_1layer = [i for i in range(int(SCALE))]
    nested_list_2layer = [[i] for i in range(int(SCALE))]
    nested_list_3layer = [[[i]] for i in range(int(SCALE))]
    nested_list_4layer = [[[[i]]] for i in range(int(SCALE))]
    nested_list_5layer = [[[[[i]]]] for i in range(int(SCALE))]
    nested_list_6layer = [[[[[[i]]]]] for i in range(int(SCALE))]
    nested_list_7layer = [[[[[[[i]]]]]] for i in range(int(SCALE))]
    nested_list_8layer = [[[[[[[[i]]]]]]] for i in range(int(SCALE))]
    nested_list_9layer = [[[[[[[[[i]]]]]]]] for i in range(int(SCALE))]
    nested_list_10layer = [[[[[[[[[[i]]]]]]]]] for i in range(int(SCALE))]

    # measure the time it takes to flatten the list
    start = time.time()
    flatten_list(nested_list_1layer)
    end = time.time()
    print(f"Time taken to flatten a 1-layer nested list with {SCALE} elements: {end - start:.6f} seconds")

    start = time.time()
    flatten_list(nested_list_2layer)
    end = time.time()
    print(f"Time taken to flatten a 2-layer nested list with {SCALE} elements: {end - start:.6f} seconds")

    start = time.time()
    flatten_list(nested_list_3layer)
    end = time.time()
    print(f"Time taken to flatten a 3-layer nested list with {SCALE} elements: {end - start:.6f} seconds")

    start = time.time()
    result = flatten_list(nested_list_4layer)
    end = time.time()
    print(f"Time taken to flatten a 4-layer nested list with {SCALE} elements: {end - start:.6f} seconds")

    start = time.time()
    flatten_list(nested_list_5layer)
    end = time.time()
    print(f"Time taken to flatten a 5-layer nested list with {SCALE} elements: {end - start:.6f} seconds")

    start = time.time()
    flatten_list(nested_list_6layer)
    end = time.time()
    print(f"Time taken to flatten a 6-layer nested list with {SCALE} elements: {end - start:.6f} seconds")

    start = time.time()
    flatten_list(nested_list_7layer)
    end = time.time()
    print(f"Time taken to flatten a 7-layer nested list with {SCALE} elements: {end - start:.6f} seconds")

    start = time.time()
    flatten_list(nested_list_8layer)
    end = time.time()
    print(f"Time taken to flatten a 8-layer nested list with {SCALE} elements: {end - start:.6f} seconds")

    start = time.time()
    flatten_list(nested_list_9layer)
    end = time.time()
    print(f"Time taken to flatten a 9-layer nested list with {SCALE} elements: {end - start:.6f} seconds")

    start = time.time()
    flatten_list(nested_list_10layer)
    end = time.time()
    print(f"Time taken to flatten a 10-layer nested list with {SCALE} elements: {end - start:.6f} seconds")
