import matplotlib.pyplot as plt
import skimage as ski
import skimage.segmentation
import skimage.util
import skimage.color
import skimage.feature
import numpy as np
import matplotlib.patches as mpatches

scale = 1.0
sigma = 0.8
min_size = 50


def _calc_colour_hist(img):
    """
        calculate colour histogram for each region
        the size of output histogram will be BINS * COLOUR_CHANNELS(3)
        number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
        extract HSV
    """
    BINS = 25
    hist = np.array([])
    for colour_channel in (0, 1, 2):
        # extracting one colour channel
        c = img[:, colour_channel]
        # calculate histogram for each colour and join to the result
        hist = np.concatenate(
            [hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]])
    # L1 normalize
    hist = hist / len(img)
    return hist


def _calc_texture_hist(img):
    """
        calculate texture histogram for each region
        calculate the histogram of gradient for each colours
        the size of output histogram will be
            BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """
    BINS = 10
    hist = np.array([])
    for colour_channel in (0, 1, 2):
        # mask by the colour channel
        fd = img[:, colour_channel]
        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = np.concatenate(
            [hist] + [np.histogram(fd, BINS, (0.0, 1.0))[0]])
    # L1 Normalize
    hist = hist / len(img)
    return hist


def intersect(a, b):
    if (a["min_x"] < b["min_x"] < a["max_x"]
        and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
            and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
            and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
            and a["min_y"] < b["min_y"] < a["max_y"]):
        return True
    return False


def _sim_colour(r1, r2):
    """
        calculate the sum of histogram intersection of colour
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def _sim_texture(r1, r2):
    """
        calculate the sum of histogram intersection of texture
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def _sim_size(r1, r2, imsize):
    """
        calculate the size similarity over the image
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
            (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
            * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def _calc_sim(r1, r2, imsize):
    return (_sim_colour(r1, r2) + _sim_texture(r1, r2)
            + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize))


def _merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


def main():
    # get seg
    img = ski.data.astronaut()
    img_msk = ski.segmentation.felzenszwalb(ski.util.img_as_float(img), scale, sigma, min_size)
    # plt.imshow(img_msk)
    # plt.show()
    img_merged = np.append(img, np.zeros(img.shape[:2])[:, :, np.newaxis], axis=2)
    img_merged[:, :, 3] = img_msk
    imsize = img_merged.shape[0] * img_merged.shape[1]

    # get regions
    R = {}
    hsv = ski.color.rgb2hsv(img)
    for y, rows in enumerate(img_merged):
        for x, (r, g, b, l) in enumerate(rows):
            # print(l)
            # l is being histogram key of max value of Felzenszwalb segmentation
            if l not in R:
                R[l] = {"min_x": 0xffff, "min_y": 0xffff, "max_x": 0, "max_y": 0, "labels": [l]}
            # bounding box
            R[l]["min_x"] = min(R[l]["min_x"], x)
            R[l]["min_y"] = min(R[l]["min_y"], y)
            R[l]["max_x"] = max(R[l]["max_x"], x)
            R[l]["max_y"] = max(R[l]["min_y"], y)
    # for i in R.keys():
    #     print(f'key is :{i},and values is : {R[i]}')
    # get gradient
    grad = np.zeros(img_merged[:, :, :3].shape)
    for c in range(3):
        grad[:, :, c] = ski.feature.local_binary_pattern(img_merged[:, :, c], 8, 1.0)

    # calculate colour histogram of each region
    for k, v in list(R.items()):
        masked_pixels = hsv[:, :, :][img_merged[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)
        R[k]["hist_t"] = _calc_texture_hist(grad[:, :][img_merged[:, :, 3] == k])

    # get neighbours
    regions = list(R.items())
    neighbours = []
    for cur, a in enumerate(regions[:-1]):
        for b in regions[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    # calculate initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    # hierarchal search
    while S != {}:
        # get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]
        # merge corresponding regions
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])
        # mark similarities for regions to be removed
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)
        # remove old similarities of related regions
        for k in key_to_delete:
            del S[k]
        # calculate similarity set with the new region
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)
    result = []
    for k, r in list(R.items()):
        result.append({
            'rect': (
                r['min_x'], r['min_y'],  #x , y
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']), # w , h
            'size': r['size'],
            'labels': r['labels']
        })

    # for i in result:
    #     print(i)
    # test
    candidates = set()
    for r in result:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])
    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle((x, y), h, w, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


if __name__ == "__main__":
    main()
