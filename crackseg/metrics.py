"""像素级 TP、FP、FN 计数，保留循环版与张量版实现。"""


def count_pixels(pred, target):
    tp, fp, fn = 0, 0, 0
    height, width = pred.shape
    for i in range(height):
        for j in range(width):
            if pred[i,j] == True and target[i,j] == True:
                tp += 1
            if pred[i,j] == True and target[i,j] ==False:
                fp += 1
            if pred[i,j] == False and target[i,j] == True:
                fn += 1
    return {"tp": tp, "fp": fp, "fn": fn}


def countspixels(pred, target):
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    return tp, fp, fn
