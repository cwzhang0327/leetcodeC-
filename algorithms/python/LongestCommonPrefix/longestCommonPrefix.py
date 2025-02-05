def longest_common_prefix(strs):
    if not strs:
        return ""
    
    prefix = strs[0]
    
    for s in strs[1:]:
        while s[:len(prefix)] != prefix and prefix:
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix

if __name__ == "__main__":
    strs = ["abab", "aba", "abc"]
    print(longest_common_prefix(strs))