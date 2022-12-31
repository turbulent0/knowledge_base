def test():
    inp = list(range(1, 21))
    i = 2
    while len(inp) > 2:
        inp.extend(inp[:i])
        inp = inp[i+1:]
    return inp[1]

print(test())