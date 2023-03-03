
banned = [1,6,5]
n = 5
maxSum = 6
def maxCount(banned, n, maxSum):
    """
    :type banned: List[int]
    :type n: int
    :type maxSum: int
    :rtype: int
    
    """
    s = set(banned)
    res = 0
    for i in range(1, n+1):
        if i not in s and maxSum-i > 0:
            res += 1
            maxCount -= i
    return res
maxCount(banned, n, maxSum)

