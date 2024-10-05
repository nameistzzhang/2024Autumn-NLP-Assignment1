import time
from submission import char_count

SCALE = 1e8

if __name__ == "__main__":
    
    # produce a string of SCALE charactors that each character is randomly selected from a-z

    print("Producing a string of SCALE characters...")
    s = ''.join([chr(ord('a') + i % 26) for i in range(int(SCALE))])

    # start timer
    start = time.time()
    char_count(s)    # call the function
    # end timer
    end = time.time()

    # print the time taken
    print(f"Time taken to count characters in a string with {SCALE} characters: {end - start:.6f} seconds")