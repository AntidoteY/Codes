def getBondPrice(y, face, couponRate, m, ppy=1):
    pvcfsum = 0
    for i in range(1, m * ppy + 1):
        pvcfsum += (face * couponRate / ppy) * (1 + (y / ppy)) ** (-i)
    pvcfsum += face * (1 + (y / ppy)) ** (-m * ppy)
    return pvcfsum
