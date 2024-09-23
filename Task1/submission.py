def flatten_list(nested_list: list):
    """
    flatten_list: Given a list of nested lists, return a single list with all the elements of the nested lists.
    Parameters:
        nested_list: list of nested lists.
    Returns:
        list: a single list with all the elements of the nested lists.
    """

    ans = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            ans.extend(flatten_list(sublist))
        else:
            ans.append(sublist)

    return ans

def char_count(s: str):
    """
    char_count: Given a string, return a dictionary with the count of each character in the string.
    Parameters:
        s: string.
    Returns:
        dict: a dictionary with the count of each character in the string.
    """

    count = {}
    for char in s:
        count[char] = count.get(char, 0) + 1
    return count