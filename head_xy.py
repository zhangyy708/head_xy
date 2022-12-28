# This program is developed by itisastart (zhangyy708 at github).
# Any individual could use this program as a reference if giving proper credits and not using for commercial use 
# (but earning money through livestreaming is allowed). 
# This program is free and open-source. All rights reserved.
# 
# Disclaimer: 
# I have learned python for like a month when writing this program... 
# so I may not be able to solve all bugs... 
# If you find a bug, feel free to send me a message at bilibili or send an email to me at zhangyy.708@gmail.com
# ... or just try to figure it out by yourself! ganbare!
# And I use only English in this file just to avoid ASCII / UTF-8 thing. 
# If you want to send me a message, Mandarin/English/Japanese are all OK!

########################################################################################################
import numpy as np
from matplotlib import pyplot as plt
import math

# images need to be layers from the same .psd file (aligned)
# there should not be any colored pixels other than specific objects
# the background should be transparent

# images are assumed to be symmetric
# open mouth (rather than closed) is recommended

skip = input('''Need to enter paramters? (Y/N) ''')
if len(skip) < 1 or skip == 'Y' or skip == 'y' :
    skip = False
elif skip == 'N' or skip == 'n' :
    skip = True
else: 
    skip = True    

if not skip :
    filename_face = input('Please enter the name of the face file: ')
    filename_eyes = input('Please enter the name of the eyes file: ')
    filename_mouth = input('Please enter the name of the mouth file: ')
    filename_hair = input('Please enter the name of the front hair file: ')
    if len(filename_face) < 1 :
        filename_face = 'xy_face.png' # default
    if len(filename_eyes) < 1 :
        filename_eyes = 'xy_eyes.png'
    if len(filename_mouth) < 1 :
        filename_mouth = 'xy_mouth.png'
    if len(filename_hair) < 1 :
        filename_hair = 'xy_hair.png'

    alpha_x = input('Please enter the rotation degree of x-axis (default 15): ')
    if len(alpha_x) < 1 :
        alpha_x = 15.0
    alpha_x = (float(alpha_x) / 180) * math.pi

    alpha_y = input('Please enter the rotation degree of y-axis (default 15): ')
    if len(alpha_y) < 1 :
        alpha_y = 15.0
    alpha_y = (float(alpha_y) / 180) * math.pi

else:
    filename_face = 'xy_face.png'
    filename_eyes = 'xy_eyes.png'
    filename_mouth = 'xy_mouth.png'
    filename_hair = 'xy_hair.png'

    alpha_x = math.pi / 12
    alpha_y = math.pi / 12
#########################################################################################################
# variables can be adjusted to make the transformed images more accurate

# read image files
I_face = plt.imread(filename_face) # assume the face is symmetrical
I_eyes = plt.imread(filename_eyes)
I_mouth = plt.imread(filename_mouth)
I_hair = plt.imread(filename_hair)

# the scale of the radius of the hair's rotation to the face's rotation
r_scale_x_hair = 1.075

# the scale of the radius of the eyes' rotation to the face's rotation
r_scale_x = 1.05

# the scale of the degree of the mouth's rotation to the face's rotation
alpha_scale_x = 1.075

# find the range of the head (up < down, left < right)
for i in range(0, I_face.shape[0]): # from top to bottom
    if I_face[i, round(I_face.shape[1] / 2), 3] != 0 : # start from the middle vertical line
        head_up = i
        break

while True :
    if head_up <= 0 : # if the pixel is already at the edge of the image
        break
    if sum(I_face[head_up - 1, :, 3]) == 0: # if the line above the current pixel is transparent
        break
    head_up = head_up - 1

for i in range(I_face.shape[0] - 1, -1, -1): # from bottom to top
    if I_face[i, round(I_face.shape[1] / 2), 3] != 0 : # start from the middle vertical line
        head_down = i
        break

while True :
    if head_down >= I_face.shape[0] - 1 : # if the pixel is already at the edge of the image
        break
    if sum(I_face[head_down + 1, :, 3]) == 0: # if the line below the current pixel is transparent
        break
    head_down = head_down + 1

head_h = head_down - head_up # the height of the head

head_left = round(I_face.shape[1] / 2) 
head_right = round(I_face.shape[1] / 2)
while True :
    if head_left <= 0 :
        break
    if sum(I_face[:, head_left - 1, 3]) == 0 :
        break
    head_left = head_left - 1

while True :
    if head_right >= I_face.shape[1] - 1 :
        break
    if sum(I_face[:, head_right + 1, 3]) == 0 :
        break
    head_right = head_right + 1

head_w = head_right - head_left # the width of the head

# the rotation radius
r = (head_w / 2) * 1

# find the ranage of images
range_up = max(0, round(head_up - head_h / 3)) # the range of the images
range_down = min(I_face.shape[0], round(head_down + head_h / 3))
range_left = max(0, round(head_left - head_w / 3))
range_right = min(I_face.shape[1], round(head_right + head_w / 3))

I_face = I_face[range_up:range_down, range_left:range_right]
I_eyes = I_eyes[range_up:range_down, range_left:range_right]
I_mouth = I_mouth[range_up:range_down, range_left:range_right]
I_hair = I_hair[range_up:range_down, range_left:range_right]

# rotation center (default: 1/3 at the lower part of the face)
(m, n) = (range_down - (range_down - head_down) - head_h / 3, (range_right - range_left) / 2) 

#########################################################################################################
# y-axis
def rotate_y(I, r, alpha_y, hair = False):

    I1 = np.zeros(I.shape, dtype = float)
    I2 = np.zeros(I.shape, dtype = float)

    for i in range(0, I.shape[0]): # vertical range
        for j in range(0, I.shape[1]): # horizontal range

            if I[i, j, 3] == 0: # if the pixel is transparent
                continue
            j1 = j
            j2 = j

            # above forehead (hair): sphere (change r)
            # under forehead: cylinder
            if hair:
                x0 = i - m
                y0 = j - n
                r_s_2 = abs(r**2  - (r - i + head_up - range_up)**2)
                z0 = math.sqrt(abs(r_s_2 - y0**2))
            else:
                # find the position of the original pixel
                x0 = i - m
                y0 = j - n
                z0 = math.sqrt(abs(r**2 - y0**2))

            # find x1 of the transferred pixel
            x1 = z0 * math.sin(alpha_y) + x0 * math.cos(alpha_y)
            i1 = max(min(int(x1 + m), I.shape[0] -1), 0)

            x2 = z0 * math.sin(-alpha_y) + x0 * math.cos(-alpha_y)
            i2 = max(min(int(x2 + m), I.shape[0] -1), 0)
            
            # color
            if I1[i1, j1].all() == 0 : # if (i1, j1) has not be colored
                I1[i1, j1] = I[i, j]
            if I2[i2, j2].all() == 0 : 
                I2[i2, j2] = I[i, j]

    return (I1, I2)

# x-axis
def rotate_x(I, r, alpha_x, k = 0, hair = False): # k should be non-positive

    I1 = np.zeros(I.shape, dtype = float)
    I2 = np.zeros(I.shape, dtype = float)

    for i in range(0, I.shape[0]): # vertical range
        for j in range(0, I.shape[1]): # horizontal range

            if I[i, j, 3] == 0: # if the pixel is transparent
                continue

            i1 = i
            i2 = i

            # above forehead (hair): sphere (change r)
            # under forehead: cylinder
            if hair:
                x0 = i - m
                y0 = j - n
                r_s_2 = abs(r**2  - (r - i + head_up - range_up)**2) * (r_scale_x_hair ** 2)
                z0 = math.sqrt(abs(r_s_2 - y0**2))
            else:
                # find the position of the original pixel
                x0 = i - m
                y0 = j - n
                z0 = math.sqrt(abs(r**2 - y0**2))

            # find x1 of the transferred pixel
            y1 = (z0 - k) * math.sin(alpha_x) + y0 * math.cos(alpha_x)
            j1 = max(min(int(y1 + n), I.shape[1] -1), 0)

            y2 = (z0 - k) * math.sin(-alpha_x) + y0 * math.cos(-alpha_x)
            j2 = max(min(int(y2 + n), I.shape[1] -1), 0)

            # color
            if I1[i1, j1].all() == 0 : # if (i1, j1) has not be colored
                I1[i1, j1] = I[i, j]
            if I2[i2, j2].all() == 0 : 
                I2[i2, j2] = I[i, j]

    return (I1, I2)

# combining x and y axes
def rotate_xy(I, r, alpha_x, alpha_y, hair = False):

    I1 = np.zeros(I.shape, dtype = float)
    I2 = np.zeros(I.shape, dtype = float)
    I3 = np.zeros(I.shape, dtype = float)
    I4 = np.zeros(I.shape, dtype = float)

    for i in range(0, I.shape[0]): # vertical range
        for j in range(0, I.shape[1]): # horizontal range

            if I[i, j, 3] == 0: # if the pixel is transparent
                continue

            # above forehead (hair): sphere (change r)
            # under forehead: cylinder
            if hair:
                r0_2 = abs(r**2  - (r - i + head_up - range_up)**2) * (r_scale_x_hair ** 2)
            else:
                r0_2 = r ** 2         

            # find the position of the original pixel
            x0 = i - m
            y0 = j - n
            z0 = math.sqrt(abs(r0_2 - y0**2))

            # find the position of the moved point
            # ↘
            x1 = x0 * math.cos(alpha_y) + z0 * math.sin(alpha_y)
            y1 = z0 * math.cos(alpha_y) * math.sin(alpha_x) \
                - x0 * math.sin(alpha_x) * math.sin(alpha_y) \
                + y0 * math.cos(alpha_x)
                
            i1 = max(min(int(x1 + m), I.shape[0] - 1), 0)
            j1 = max(min(int(y1 + n), I.shape[1] - 1), 0)
            
            # ↙
            x2 = x0 * math.cos(alpha_y) + z0 * math.sin(alpha_y)
            y2 = z0 * math.cos(alpha_y) * math.sin(-alpha_x) \
                - x0 * math.sin(-alpha_x) * math.sin(alpha_y) \
                + y0 * math.cos(-alpha_x)
                
            i2 = max(min(int(x2 + m), I.shape[0] - 1), 0)
            j2 = max(min(int(y2 + n), I.shape[1] - 1), 0)

            # ↗
            x3 = x0 * math.cos(-alpha_y) + z0 * math.sin(-alpha_y)
            y3 = z0 * math.cos(-alpha_y) * math.sin(alpha_x) \
                - x0 * math.sin(alpha_x) * math.sin(-alpha_y) \
                + y0 * math.cos(alpha_x)
                
            i3 = max(min(int(x3 + m), I.shape[0] - 1), 0)
            j3 = max(min(int(y3 + n), I.shape[1] - 1), 0)
            
            # ↖
            x4 = x0 * math.cos(-alpha_y) + z0 * math.sin(-alpha_y)
            y4 = z0 * math.cos(-alpha_y) * math.sin(-alpha_x) \
                - x0 * math.sin(-alpha_x) * math.sin(-alpha_y) \
                + y0 * math.cos(-alpha_x)
                
            i4 = max(min(int(x4 + m), I.shape[0] - 1), 0)
            j4 = max(min(int(y4 + n), I.shape[1] - 1), 0)

            # color
            if I1[i1, j1].all() == 0 : # if (i1, j1) has not be colored
                I1[i1, j1] = I[i, j]
            if I2[i2, j2].all() == 0 : 
                I2[i2, j2] = I[i, j]
            if I3[i3, j3].all() == 0 : 
                I3[i3, j3] = I[i, j]
            if I4[i4, j4].all() == 0 : 
                I4[i4, j4] = I[i, j]

    return (I1, I2, I3, I4)

########################################################################################
# draw
def cmp_images(r, alpha_x, alpha_y, r_scale_x, alpha_scale_x):

    print('Printing now: ')
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(3, 3, 5)
    ax.imshow(I_face)
    ax.imshow(I_eyes)
    ax.imshow(I_mouth)
    ax.imshow(I_hair)
    print('Position 5 done.')

    # ←→
    (I_face_r6, I_face_r4) = rotate_x(I_face, r, alpha_x)
    # (I_eyes_r6, I_eyes_r4) = rotate_x(I_eyes, r * r_scale_x, alpha_x, k = r * (r_scale_x - 1))
    (I_eyes_r6, I_eyes_r4) = rotate_x(I_eyes, r * r_scale_x, alpha_x)
    (I_mouth_r6, I_mouth_r4) = rotate_x(I_mouth, r, alpha_x * alpha_scale_x)
    (I_hair_r6, I_hair_r4) = rotate_x(I_hair, r * r_scale_x_hair, alpha_x, hair = True)
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.imshow(I_face_r4)
    ax4.imshow(I_eyes_r4)
    ax4.imshow(I_mouth_r4)
    ax4.imshow(I_hair_r4)
    print('Position 4 done.')

    ax6 = fig.add_subplot(3, 3, 6)
    ax6.imshow(I_face_r6)
    ax6.imshow(I_eyes_r6)
    ax6.imshow(I_mouth_r6)
    ax6.imshow(I_hair_r6)
    print('Position 6 done.')

    # ↑↓
    (I_face_r8, I_face_r2) = rotate_y(I_face, r, alpha_y)
    (I_eyes_r8, I_eyes_r2) = rotate_y(I_eyes, r * r_scale_x, alpha_y)
    (I_mouth_r8, I_mouth_r2) = rotate_y(I_mouth, r, alpha_y)
    (I_hair_r8, I_hair_r2) = rotate_y(I_hair, r * r_scale_x_hair, alpha_y, hair = True)
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.imshow(I_face_r2)
    ax2.imshow(I_eyes_r2)
    ax2.imshow(I_mouth_r2)
    ax2.imshow(I_hair_r2)
    print('Position 2 done.')

    ax8 = fig.add_subplot(3, 3, 8)
    ax8.imshow(I_face_r8)
    ax8.imshow(I_eyes_r8)
    ax8.imshow(I_mouth_r8)
    ax8.imshow(I_hair_r8)
    print('Position 8 done.')
    
    # # the following codes are abandoned, but it is possible to use them (instead of rotate_xy()) if needed
    # # temp ←→
    # (I_face_temp6, I_face_temp4) = \
    #     rotate_x(I_face, r, alpha_x * math.sqrt(2)/2)
    # (I_eyes_temp6, I_eyes_temp4) = \
    #     rotate_x(I_eyes, r * r_scale_x, alpha_x * math.sqrt(2)/2, k = r * (r_scale_x - 1))
    # (I_mouth_temp6, I_mouth_temp4) = \
    #     rotate_x(I_mouth, r, alpha_x * alpha_scale_x * math.sqrt(2)/2, k = 0)
    # (I_hair_temp6, I_hair_temp4) = \
    #     rotate_x(I_hair, r * r_scale_x_hair, alpha_x * math.sqrt(2)/2, hair = True)

    # # ↙↖
    # (I_face_r7, I_face_r1) = rotate_y(I_face_temp4, r, alpha_y * math.sqrt(2)/2)
    # (I_eyes_r7, I_eyes_r1) = rotate_y(I_eyes_temp4, r * r_scale_x, alpha_y * math.sqrt(2)/2)
    # (I_mouth_r7, I_mouth_r1) = rotate_y(I_mouth_temp4, r, alpha_y * math.sqrt(2)/2)
    # (I_hair_r7, I_hair_r1) = rotate_y(I_hair_temp4, r * r_scale_x_hair, alpha_y * math.sqrt(2)/2, hair = True)

    # # ↘↗
    # (I_face_r9, I_face_r3) = rotate_y(I_face_temp6, r, alpha_y * math.sqrt(2)/2)
    # (I_eyes_r9, I_eyes_r3) = rotate_y(I_eyes_temp6, r * r_scale_x, alpha_y * math.sqrt(2)/2)
    # (I_mouth_r9, I_mouth_r3) = rotate_y(I_mouth_temp6, r, alpha_y * math.sqrt(2)/2) 
    # (I_hair_r9, I_hair_r3) = rotate_y(I_hair_temp6, r * r_scale_x_hair, alpha_y * math.sqrt(2)/2, hair = True)

    (I_face_r9, I_face_r7, I_face_r3, I_face_r1) = rotate_xy(I_face, r, alpha_x, alpha_y)
    (I_eyes_r9, I_eyes_r7, I_eyes_r3, I_eyes_r1) = rotate_xy(I_eyes, r * r_scale_x, alpha_x, alpha_y)
    (I_mouth_r9, I_mouth_r7, I_mouth_r3, I_mouth_r1) = rotate_xy(I_mouth, r, alpha_x, alpha_y)
    (I_hair_r9, I_hair_r7, I_hair_r3, I_hair_r1) = rotate_xy(I_hair, r * r_scale_x_hair, alpha_x, alpha_y, hair = True)

    ax1 = fig.add_subplot(3, 3, 1)
    ax1.imshow(I_face_r1)
    ax1.imshow(I_eyes_r1)
    ax1.imshow(I_mouth_r1)
    ax1.imshow(I_hair_r1)
    print('Position 1 done.')

    ax3 = fig.add_subplot(3, 3, 3)
    ax3.imshow(I_face_r3)
    ax3.imshow(I_eyes_r3)
    ax3.imshow(I_mouth_r3)
    ax3.imshow(I_hair_r3)
    print('Position 3 done.')

    ax7 = fig.add_subplot(3, 3, 7)
    ax7.imshow(I_face_r7)
    ax7.imshow(I_eyes_r7)
    ax7.imshow(I_mouth_r7)
    ax7.imshow(I_hair_r7)
    print('Position 7 done.')

    ax9 = fig.add_subplot(3, 3, 9)
    ax9.imshow(I_face_r9)
    ax9.imshow(I_eyes_r9)
    ax9.imshow(I_mouth_r9)
    ax9.imshow(I_hair_r9)
    print('Position 9 done.')

    plt.show()

cmp_images(r, alpha_x, alpha_y, r_scale_x, alpha_scale_x)
