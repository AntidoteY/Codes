def FizzBuzz(start, finish):
    outlist = []
    # <Your work>
    for number in range(start, finish+1):
        if number % 15 == 0:
            outlist.append("fizzbuzz")
        elif number % 3 == 0:
            outlist.append("fizz")
        elif number % 5 == 0:
            outlist.append("buzz")
        else:
            outlist.append(number)
    return outlist
