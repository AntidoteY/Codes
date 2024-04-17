def getBondDuration(y, face, couponRate, m, ppy=1):
    pvcfsum = 0
    duration = 0
    coupon = face * couponRate/ppy
    for i in range(1, m*ppy + 1):
        pvcf = coupon * (1 + y/ppy) ** (-i)
        pvcfsum += pvcf
        duration += pvcf * (i/ppy)
    pvcfsum += face * (1 + y/ppy) ** (-m*ppy)
    duration += (m * face) * (1 + y/ppy) ** (-m*ppy)
    duration = duration/pvcfsum

    return duration
