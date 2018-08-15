#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import cv2
import numba
import numpy as np
from numba import prange, njit

mem = {}
cols = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (128, 128, 0), 4: (0, 0, 128), 5: (128, 0, 128),
        6: (0, 128, 128), 7: (192, 192, 192), 8: (128, 128, 128), 9: (255, 0, 0), 10: (0, 255, 0), 11: (255, 255, 0),
        12: (0, 0, 255), 13: (255, 0, 255), 14: (0, 255, 255), 15: (255, 255, 255), 16: (0, 0, 0),
        17: (0, 0, 95), 18: (0, 0, 135), 19: (0, 0, 175), 20: (0, 0, 215), 21: (0, 0, 255), 22: (0, 95, 0),
        23: (0, 95, 95), 24: (0, 95, 135), 25: (0, 95, 175), 26: (0, 95, 215),
        27: (0, 95, 255), 28: (0, 135, 0), 29: (0, 135, 95), 30: (0, 135, 135), 31: (0, 135, 175), 32: (0, 135, 215),
        33: (0, 135, 255), 34: (0, 175, 0), 35: (0, 175, 95),
        36: (0, 175, 135), 37: (0, 175, 175), 38: (0, 175, 215), 39: (0, 175, 255), 40: (0, 215, 0), 41: (0, 215, 95),
        42: (0, 215, 135), 43: (0, 215, 175), 44: (0, 215, 215),
        45: (0, 215, 255), 46: (0, 255, 0), 47: (0, 255, 95), 48: (0, 255, 135), 49: (0, 255, 175), 50: (0, 255, 215),
        51: (0, 255, 255), 52: (95, 0, 0), 53: (95, 0, 95),
        54: (95, 0, 135), 55: (95, 0, 175), 56: (95, 0, 215), 57: (95, 0, 255), 58: (95, 95, 0), 59: (95, 95, 95),
        60: (95, 95, 135), 61: (95, 95, 175), 62: (95, 95, 215),
        63: (95, 95, 255), 64: (95, 135, 0), 65: (95, 135, 95), 66: (95, 135, 135), 67: (95, 135, 175),
        68: (95, 135, 215), 69: (95, 135, 255), 70: (95, 175, 0),
        71: (95, 175, 95), 72: (95, 175, 135), 73: (95, 175, 175), 74: (95, 175, 215), 75: (95, 175, 255),
        76: (95, 215, 0), 77: (95, 215, 95), 78: (95, 215, 135),
        79: (95, 215, 175), 80: (95, 215, 215), 81: (95, 215, 255), 82: (95, 255, 0), 83: (95, 255, 95),
        84: (95, 255, 135), 85: (95, 255, 175), 86: (95, 255, 215), 87: (95, 255, 255), 88: (135, 0, 0),
        89: (135, 0, 95), 90: (135, 0, 135), 91: (135, 0, 175), 92: (135, 0, 215), 93: (135, 0, 255), 94: (135, 95, 0),
        95: (135, 95, 95),
        96: (135, 95, 135), 97: (135, 95, 175), 98: (135, 95, 215), 99: (135, 95, 255), 100: (135, 135, 0),
        101: (135, 135, 95), 102: (135, 135, 135), 103: (135, 135, 175), 104: (135, 135, 215), 105: (135, 135, 255),
        106: (135, 175, 0), 107: (135, 175, 95), 108: (135, 175, 135), 109: (135, 175, 175), 110: (135, 175, 215),
        111: (135, 175, 255),
        112: (135, 215, 0), 113: (135, 215, 95), 114: (135, 215, 135), 115: (135, 215, 175), 116: (135, 215, 215),
        117: (135, 215, 255), 118: (135, 255, 0), 119: (135, 255, 95),
        120: (135, 255, 135), 121: (135, 255, 175), 122: (135, 255, 215), 123: (135, 255, 255), 124: (175, 0, 0),
        125: (175, 0, 95), 126: (175, 0, 135), 127: (175, 0, 175),
        128: (175, 0, 215), 129: (175, 0, 255), 130: (175, 95, 0), 131: (175, 95, 95), 132: (175, 95, 135),
        133: (175, 95, 175), 134: (175, 95, 215), 135: (175, 95, 255),
        136: (175, 135, 0), 137: (175, 135, 95), 138: (175, 135, 135), 139: (175, 135, 175), 140: (175, 135, 215),
        141: (175, 135, 255), 142: (175, 175, 0), 143: (175, 175, 95),
        144: (175, 175, 135), 145: (175, 175, 175), 146: (175, 175, 215), 147: (175, 175, 255), 148: (175, 215, 0),
        149: (175, 215, 95), 150: (175, 215, 135), 151: (175, 215, 175),
        152: (175, 215, 215), 153: (175, 215, 255), 154: (175, 255, 0), 155: (175, 255, 95), 156: (175, 255, 135),
        157: (175, 255, 175), 158: (175, 255, 215), 159: (175, 255, 255),
        160: (215, 0, 0), 161: (215, 0, 95), 162: (215, 0, 135), 163: (215, 0, 175), 164: (215, 0, 215),
        165: (215, 0, 255), 166: (215, 95, 0), 167: (215, 95, 95),
        168: (215, 95, 135), 169: (215, 95, 175), 170: (215, 95, 215), 171: (215, 95, 255), 172: (215, 135, 0),
        173: (215, 135, 95), 174: (215, 135, 135),
        175: (215, 135, 175), 176: (215, 135, 215), 177: (215, 135, 255), 178: (215, 175, 0), 179: (215, 175, 95),
        180: (215, 175, 135), 181: (215, 175, 175),
        182: (215, 175, 215), 183: (215, 175, 255), 184: (215, 215, 0), 185: (215, 215, 95), 186: (215, 215, 135),
        187: (215, 215, 175), 188: (215, 215, 215),
        189: (215, 215, 255), 190: (215, 255, 0), 191: (215, 255, 95), 192: (215, 255, 135), 193: (215, 255, 175),
        194: (215, 255, 215), 195: (215, 255, 255),
        196: (255, 0, 0), 197: (255, 0, 95), 198: (255, 0, 135), 199: (255, 0, 175), 200: (255, 0, 215),
        201: (255, 0, 255), 202: (255, 95, 0), 203: (255, 95, 95),
        204: (255, 95, 135), 205: (255, 95, 175), 206: (255, 95, 215), 207: (255, 95, 255), 208: (255, 135, 0),
        209: (255, 135, 95), 210: (255, 135, 135), 211: (255, 135, 175),
        212: (255, 135, 215), 213: (255, 135, 255), 214: (255, 175, 0), 215: (255, 175, 95), 216: (255, 175, 135),
        217: (255, 175, 175), 218: (255, 175, 215), 219: (255, 175, 255),
        220: (255, 215, 0), 221: (255, 215, 95), 222: (255, 215, 135), 223: (255, 215, 175), 224: (255, 215, 215),
        225: (255, 215, 255), 226: (255, 255, 0), 227: (255, 255, 95),
        228: (255, 255, 135), 229: (255, 255, 175), 230: (255, 255, 215), 231: (255, 255, 255), 232: (8, 8, 8),
        233: (18, 18, 18), 234: (28, 28, 28), 235: (38, 38, 38),
        236: (48, 48, 48), 237: (58, 58, 58), 238: (68, 68, 68), 239: (78, 78, 78), 240: (88, 88, 88),
        241: (98, 98, 98), 242: (108, 108, 108), 243: (118, 118, 118),
        244: (128, 128, 128), 245: (138, 138, 138), 246: (148, 148, 148), 247: (158, 158, 158), 248: (168, 168, 168),
        249: (178, 178, 178), 250: (188, 188, 188), 251: (198, 198, 198), 252: (208, 208, 208), 253: (218, 218, 218),
        254: (228, 228, 228), 255: (238, 238, 238), }.items()

# cols_bgr = np.array([y[::-1] for x, y in cols])
cols_bgr = np.empty((256, 3), dtype=np.int64)
for x, y in cols:
    cols_bgr[x] = y[::-1]

cols_4bit_items = {30: (1, 1, 1),
                   31: (222, 56, 43),
                   32: (57, 181, 74),
                   33: (255, 199, 6),
                   34: (0, 111, 184),
                   35: (118, 38, 113),
                   36: (44, 181, 233),
                   37: (204, 204, 204),
                   90: (128, 128, 128),
                   91: (255, 0, 0),
                   92: (0, 255, 0),
                   93: (255, 255, 0),
                   94: (0, 0, 255),
                   95: (255, 0, 255),
                   96: (0, 255, 255),
                   97: (255, 255, 255)}.items()


def closest_col(pxl, palette=cols_bgr):
    tpl = tuple(pxl)
    if tpl in mem:
        return mem[tpl]
    out = str(np.argmin(dists(palette, pxl)))
    mem[tpl] = out
    return out


@njit(parallel=True)
def dists(col_map, pxl):
    dists = np.empty(col_map.shape[0], dtype=np.double)
    for i in prange(col_map.shape[0]):
        dists[i] = col_dist(col_map[i], pxl)
    return dists


@numba.njit(fastmath=True)
def col_dist(a, b):
    r = (a[0] + b[0]) / 2
    dr = np.power(a[0] - b[0], 2)
    dg = np.power(a[1] - b[1], 2)
    db = np.power(a[2] - b[2], 2)
    return np.sqrt(2 * dr + 4 * dg + 3 * db + ((r * (dr - db)) / 256))


@numba.njit(fastmath=True)
def col_dist2(a, b):
    dr = math.pow(a[0] - b[0], 2)
    dg = math.pow(a[1] - b[1], 2)
    db = math.pow(a[2] - b[2], 2)
    return math.sqrt(2 * dr + 4 * dg + 3 * db)


def closest_col_4bit(pxl):
    tpl = tuple(pxl)
    if tpl in mem:
        return mem[tpl]
    out = min(cols_4bit_items, key=lambda x: col_dist(pxl, x[1][::-1]))[0]
    mem[tpl] = out
    return out


def img_4bit(input_img, height, width):
    out = []
    input_img = input_img.astype(np.int64)
    for y in range(height // 2):
        y2 = 2 * y
        for _x in range(width):
            top_pxl = input_img[y2, _x]
            bot_pxl = input_img[y2 + 1, _x]
            # get the closest colour to the pixel
            top_col = closest_col_4bit(top_pxl)
            bot_col = closest_col_4bit(bot_pxl)
            out.append(''.join(("\x1B[", str(top_col), ";", str(bot_col + 10), "m▀")))
        out.append('\n')
    return ''.join(out)


def img_24bit(input_img, height, width):
    # 48 chars per pixel pair
    # out = np.empty((height * width + 1), dtype=np.object)
    out = []
    for y in range(height // 2):
        y2 = 2 * y
        for x in range(width):
            top_pxl = input_img[y2, x]
            bot_pxl = input_img[y2 + 1, x]
            # Render the colour directly
            out.append(
                "\x1B[38;2;" + str(top_pxl[2]) + ';' + str(top_pxl[1]) + ';' + str(top_pxl[0]) + "m\x1B[48;2;" + str(
                    bot_pxl[2]) + ';' + str(bot_pxl[1]) + ';' + str(bot_pxl[0]) + "m▀")
        out.append('\n')
    return ''.join(out)


def img_8bit(input_img, height, width):
    out = []
    input_img = input_img.astype(np.int64)
    for y in range(height // 2):
        y2 = 2 * y
        for x in range(width):
            top_pxl = input_img[y2, x]
            bot_pxl = input_img[y2 + 1, x]
            # get the closest colour to the pixel
            top_col = closest_col(top_pxl)
            bot_col = closest_col(bot_pxl)
            out.append(''.join(("\x1B[38;5;", top_col, ";48;5;", bot_col, "m▀")))
        out.append('\n')
    return ''.join(out)


def fast_setup():
    import pyopencl as cl
    import pyopencl.cltypes

    device = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([device])
    lut = np.zeros(256, cl.cltypes.char3)
    for idx, col in cols_bgr:
        lut[idx][0] = col[0]
        lut[idx][1] = col[1]
        lut[idx][2] = col[2]
    g_lut = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=lut)
    krnl = """
    __kernel void closest(__global uchar4 *img,
                          __global ushort *out,
                          __constant uchar4 *lut,
                          ushort const cols) {
        int x = get_global_id(0);
        int height = get_global_size(0);
        int width = get_global_size(1);
        int y = get_global_id(1);
        int index = y * height + x;
        float dmin = 99999;
        int h = -1;
        for(int i = 0; i < cols; i++) {
            float d = hypot(img[index], lut[i]);
            if(d < dmin) {
                dmin = d;
                h = i
            }
        }
        out[index] = h;
    }
    """
    prog = cl.Program(ctx, krnl).build()
    func = prog.closest
    queue = cl.CommandQueue(ctx)
    return func, queue, ctx, g_lut


def img_fast(input_img, height, width, func, queue, ctx, g_lut):
    import pyopencl as cl
    g_img = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=input_img)
    out = np.zeros((width * height, 4), cl.cltypes.ushort)
    g_out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out.nbytes)
    func(queue, (height, width), None, g_img, g_out, g_lut, 256)

    cl.enqueue_copy(queue, out, g_out).wait()
    return img.reshape((width, height, 4))[:, :, :3]


def get_new_size(my_w, pxls):
    r = (my_w / float(pxls.shape[1]))
    hsize = int(pxls.shape[0] * r)
    return my_w, hsize


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Display image to terminal')
    parser.add_argument('-img', help='Image file to display', default=None)
    parser.add_argument('-width', default=78, help='Character width of output', type=int)
    parser.add_argument('-vid', help='Show video, default is usb camera', default='')
    parser.add_argument('-col', help='Colour scheme to use', choices=[4, 8, 24], default=8, type=int)
    args = parser.parse_args()
    fname = args.img
    print("\x1b[2J")
    func = {4: img_4bit, 8: img_8bit, 24: img_24bit}[args.col]
    # print("\x1b[2J")
    my_width = args.width
    if fname:
        image = cv2.imread(fname)
        new_size = get_new_size(my_width, image)
        image = cv2.resize(src=image, dsize=new_size)
        chars = func(image, new_size[1], new_size[0])
        print("\x1b[;H", chars, '\x1b[0m', sep='')

    else:
        if args.vid == '':
            cam = cv2.VideoCapture(0)
        else:
            cam = cv2.VideoCapture(args.vid)

        from time import time

        start_time = time()
        count = 0
        retval, image = cam.read()
        new_size = get_new_size(my_width, image)
        while 1:
            if not cam.isOpened():
                break
            try:
                retval, image = cam.read()
                image = cv2.resize(src=image, dsize=new_size)
                print("\x1b[;H", func(image, new_size[1], new_size[0]), '\x1B[0m', sep='')
                print("FPS:", count / (time() - start_time))
                count += 1

            except KeyboardInterrupt:
                print("FPS:", count / (time() - start_time))
                break
        cam.release()
