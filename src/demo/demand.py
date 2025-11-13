def demand(time: float, t1: float, t2: float, end: float, max: float) -> float:
    if time < t1:
        return time * max / t1
    elif time > end:
        return 0.0
    elif time > t2:
        return max - max * (time - t2) / (end - t2)
    else:
        return max
