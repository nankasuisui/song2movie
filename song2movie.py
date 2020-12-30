import numpy as np
import librosa,pydub
import cv2 as cv
from PIL import ImageFont, ImageDraw, Image
from pydub import AudioSegment
import subprocess,os
import glob
import argparse
import copy
from math import ceil

def choice(choicelist,default_idx,message):
    print(message)
    print("choose from:")
    for i,ch in enumerate(choicelist):
        print("{}: {}".format(i,ch))
    while (1):
        cho = input()
        if cho == "":
            cho = default_idx
            break
        cho = int(cho)
        if not cho in range(len(choicelist)):
            print("invalid value.")
        else:
            break
    return choicelist[cho]        

def inp(message,rettype,default_value):
    print(message)
    i = input()
    if i == "":
        i = default_value
    else:
        i = rettype(i)
    return i

#arg
parser = argparse.ArgumentParser()
parser.add_argument("input",help="input file:")
parser.add_argument("--image",help="bg image file(should have 19:9 resolution):")
args = parser.parse_args()
song = args.input

if not (os.path.isfile(song)):
    print("invalid input file.")
    exit()

title = os.path.splitext(os.path.basename(song))[0]

'''
def deprecated():
    song = glob.glob("in/*.wav")
    if len(song) == 0:
        song = glob.glob("in/*.mp3")
        if len(song) == 0:
            song = glob.glob("in/*.ogg")
    song = song[0]
'''

fn = inp("filename(default:filename):",str,title)
video = "out/"+str(fn)+".mp4"

fps = inp("FPS(default:30):",float,30.0) #:optional
size = choice(["1080p","720p","480p"],0,"resolution:")
title = inp("title(default:filename):",str,title) #:optional
artist = inp("artist:",str,"") #:optional
start = inp("start:",int,0) #second:optional
length = inp("length(sec):",int,0) #second:optional
fadein = inp("fadein(if need,type 1):",bool,False) #:optional
fadeout = inp("fadeout(if need,type 1):",bool,False)
fade_time = inp("fadetime(sec):",int,4) #second:optional
color = choice(["black","white"],0,"main color:") #choice: black white
sfig = choice(["line","rect","None"],0,"spectrum figure:") #choice: rect line
fontpath = choice(["fonts/TsukushiAMaruGothic.ttc","fonts/M-NijimiMincho/幻ノにじみ明朝.otf"],0,"font")

#define functions
def sigmoid(x,a):
    return 1.0 / (1.0 + np.exp(-x*a))

def norm(t,start,end,nstart,nend):
    return (t - start) * (nend - nstart) / (end - start) + nstart

def sig(t,start,end,sigrange,a=1): #start,end = timeのstart,end / sigrange = 正規化範囲(+-sigrangeになる)
    nt = norm(t,start,end,-sigrange,sigrange)
    return sigmoid(nt,a)

def puttext_ja(cv_image, text, point, font_path, font_size, color):
    font = ImageFont.truetype(font_path, font_size)

    cv_rgb_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_rgb_image)

    draw = ImageDraw.Draw(pil_image)
    draw.text(point, text, fill=color, font=font)

    cv_rgb_result_image = np.asarray(pil_image)
    cv_bgr_result_image = cv.cvtColor(cv_rgb_result_image, cv.COLOR_RGB2BGR)

    return cv_bgr_result_image

def resize_trim(img, target_width, target_height):
    h,w = img.shape[:2]
    scale_w = float(target_width) / w
    scale_h = float(target_height) / h
    scale = max([scale_w,scale_h])
    x = ceil(scale * w)
    y = ceil(scale * h)
    #img = cv.resize(img,fx=scale,fy=scale,dsize=(0,0))
    imgg = cv.resize(img,dsize=(x,y))
    return imgg[0:target_height,0:target_width]

    
#load audio file
try:
    dubclip = AudioSegment.from_file(song,format=song[-3:])
except Exception as e:
    print("unexpected song format.",song[-3:])
    exit()

#check duration >= length
if length != 0 and dubclip.duration_seconds < length:
    print("song is shorter than expected length.")
    exit()

if length == 0:
    print("length not specified. use song duration.")
    length = dubclip.duration_seconds - start
    
    
#clip audio
dubclip = dubclip[start*1000:]
dubclip = dubclip[:length*1000]

#fade audio
if fadein:dubclip = dubclip.fade(from_gain=-np.inf, duration=fade_time*1000, start=0)
if fadeout:dubclip = dubclip.fade(to_gain=-np.inf, duration=fade_time*1000, end=float('inf'))

dubclip.export("tmp.wav",format="wav")

#load spectrums
y,sr = librosa.load("tmp.wav",sr=None,mono=False)
y = librosa.core.to_mono(y) #from stereo to mono for simplicity
y = librosa.util.normalize(y)
spec = librosa.feature.melspectrogram(y=y,sr=sr,hop_length=int(sr*1/fps),power=2.0,norm=1,n_mels=64)
spec = librosa.power_to_db(spec)

#normalize spectrums
spec = norm(spec,np.min(spec),np.max(spec),0,255).astype(np.uint8)
#plt.imshow(spec,origin='lower')


#draw
drawsize = {"720p":(1280,720),"1080p":(1920,1080),"480p":(640,480)}
#fourcc = cv.VideoWriter_fourcc("X","2","6","4")
writer = cv.VideoWriter("tmp.avi",0,fps,drawsize[size])

bg_bk = lambda: np.zeros((drawsize["1080p"][1],drawsize["1080p"][0],3),dtype=np.uint8)
bg_wh = lambda: np.full((drawsize["1080p"][1],drawsize["1080p"][0],3),255,dtype=np.uint8)

col_bk = (0,0,0)
col_wh = (255,255,255)

if color == "black":
    bg_ = bg_bk
    col_ = col_wh
elif color == "white":
    bg_ = bg_wh
    col_ = col_bk

if args.image:
    img = cv.imread(args.image)
    if img is None:
        print("image can't be loaded.")
    else:
        img = resize_trim(img,drawsize["1080p"][0],drawsize["1080p"][1])
        bg_ = lambda: copy.deepcopy(img)


for f in range(int(length*fps)):
    
    print("rendering: {} / {}".format(f,int(length*fps)),end="\r")
    #background
    frame = bg_()
    
    #spectrum
    if sfig == "rect":
        #ver.rectangle
        for i,n in enumerate(spec[:,f]):
            for h in range(int(norm(n,0,255,0,24))):
                cv.rectangle(frame, (80+i*28,800-h*20),(80+i*28+14,800-h*20-14),col_, -1, cv.LINE_AA)
                
    elif sfig == "line":
        #ver.line
        for i,n in enumerate(spec[:,f]):
            cv.line(frame,(80+i*28,800),(80+i*28,800-int(2.7*n)),col_,10,cv.LINE_AA)
    
    #text
    #cv.putText(frame,title,(80,980),cv.FONT_HERSHEY_SIMPLEX,4,col_,4,cv.LINE_AA)
    #alter
    ## title
    frame = puttext_ja(frame,title,(80,840),fontpath,130,col_)
    ## artist
    frame = puttext_ja(frame,artist,(100,980),fontpath,70,col_)

    #fadein
    if False:#fadein:
        if f < fade_time*fps:
            alpha = sig(f,0,fade_time*fps,2,2) if f != 0 else 0
            frame = cv.addWeighted(frame,alpha,bg_(),1.0-alpha,0)
    
    #fadeout
    if fadeout:
        if f >= length*fps-1 - fade_time*fps:
            alpha = sig(f,length*fps-1,length*fps-1 - fade_time*fps,2,2) if f != length*fps-1 else 0
            frame = cv.addWeighted(frame,alpha,bg_(),1.0-alpha,0)
        
    #resize
    if size == "1080p":
        pass
    elif size == "720p":
        frame = cv.resize(frame,(1280,720))

    writer.write(frame)
    
    
writer.release()


#save
cmd = "./ffmpeg -y -i {} -i {} {} {} -flags global_header {}".format("tmp.avi","tmp.wav","-vcodec h264_videotoolbox -pix_fmt yuv420p -r",fps,video)
#-vcodec copy -acodec copy  
#print(cmd)
resp = subprocess.call(cmd, shell=True)
#os.remove("tmp.avi")
os.remove("tmp.wav")
