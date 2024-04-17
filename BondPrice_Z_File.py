def getBondPrice_Z(face, couponRate, times, yc):
    coupoon = face*couponRate
    pvcfsum = 0
    for time, ytm in zip(times, yc):
        pvcf = coupoon*(1+ytm)**-time
        pvcfsum += pvcf
    if times:
        pvcfsum += face*(1+yc[-1])**-times[-1]
    return pvcfsum
