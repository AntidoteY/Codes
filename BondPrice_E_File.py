def getBondPrice_E(face, couponRate, yc):
    coupon = face * couponRate
    pvcfsum = 0
    for year, ytm in enumerate(yc, start=1):
        pvcf = coupon * (1 + ytm) ** -year
        pvcfsum += pvcf
    pvcfsum += face * (1 + yc[-1]) ** -len(yc)
    return pvcfsum
