from naoqi import ALProxy
import math
import motion
from PIL import Image
import qi
import argparse
import sys
import time



def take_image(session, number):
    """
    First get an image, then show it on the screen with PIL.
    """
    # Get the service ALVideoDevice.
    
    video_service = session.service("ALVideoDevice")
    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")

    #resolution : 2 for 640x480, 3 for 1280x960, 4 for 2560x1920

    resolution = 2    # VGA 640x480
    colorSpace = 11   # RGB

    motion_service.setStiffnesses("Head", 1.0)
    motion_service.setStiffnesses("HeadYaw", 1.0)
    motion_service.setStiffnesses("HeadPitch", 1.0)
    motion_service.setStiffnesses("Body", 1.0)

    # wake up the robot and set him in a default position
    motion_service.wakeUp()
    posture_service.goToPosture("StandInit", 0.5)


    # make sure head is in a default position
    names  = ["HeadYaw", "HeadPitch"]
    angles = [0.0, 0.0]
    times  = [1.0, 1.0]
    isAbsolute = True
    motion_service.angleInterpolation(names, angles, times, isAbsolute)

    # move slowly the head to look down. The motion will
    # take 3 seconds
    # HeadYaw -> left / right
    # HeadPitch -> up / down
    names  = "HeadPitch"
    angles = math.pi/20.0
    times  = 3.0
    isAbsolute = True
    motion_service.angleInterpolation(names, angles, times, isAbsolute, _async=True)

    # wait 4 seconds to be sure that the movement has finished
    time.sleep(4.0)


    # now we have a head looking slightly down and can proceed to take an image
    
    
    videoClient = video_service.subscribe("python_client", resolution, colorSpace, 5)

    t0 = time.time()

    # Get a camera image.
    # image[6] contains the image data passed as an array of ASCII chars.
    print "getting image" 
    naoImage = video_service.getImageRemote(videoClient)
    print "got image"
    t1 = time.time()

    # Time the image transfer.
    print "acquisition delay ", t1 - t0

    video_service.unsubscribe(videoClient)


    # Now we work with the image returned and save it as a PNG  using ImageDraw
    # package.

    # Get the image size and pixel array.
    imageWidth = naoImage[0]
    imageHeight = naoImage[1]
    array = naoImage[6]
    image_string = str(bytearray(array))
    # Create a PIL Image from our pixel array.
    im = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)
    # Save the image.
    im.show()
    resize = im.resize((100, 100))
    resize.save("C:/Users/jozef/OneDrive/Desktop/FYP_images/Images/test/camImage" + str(number) + ".png", "PNG")
    names  = ["HeadYaw", "HeadPitch"]
    angles = [0.0, 0.0]
    times  = [1.0, 1.0]
    isAbsolute = True
    motion_service.angleInterpolation(names, angles, times, isAbsolute)




def call_image(number):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.0.0.64",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    take_image(session, number)

def recognize_yes_no():
    data=[]
    ip = "10.0.0.64"
    asr = ALProxy("ALSpeechRecognition", ip, 9559)

    asr.pause(True)
    asr.setLanguage("English")


    vocabulary = ["yes", "no"]


    asr.setVocabulary(vocabulary, False)
    asr.subscribe(ip)
    memProxy = ALProxy("ALMemory", ip, 9559)
    memProxy.subscribeToEvent('WordRecognized',ip,'wordRecognized')

    asr.pause(False)

    data =  memProxy.getData("WordRecognized")

    prev = data[1]

    while True:
        data = memProxy.getData("WordRecognized")
        prob = data[1]
        if prob != prev:
            prev = prob
            print("data: %s" % data[0])
            break
    asr.unsubscribe(ip)
    return data[0]


def recognize_1_10():
    data=[]
    ip = "10.0.0.64"
    asr = ALProxy("ALSpeechRecognition", ip, 9559)

    asr.pause(True)
    asr.setLanguage("English")


    vocabulary = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

    print("ready")
    asr.setVocabulary(vocabulary, False)
    asr.subscribe(ip)
    memProxy = ALProxy("ALMemory", ip, 9559)
    memProxy.subscribeToEvent('WordRecognized',ip,'wordRecognized')

    asr.pause(False)

    data =  memProxy.getData("WordRecognized")

    prev = data[1]

    while True:
        data = memProxy.getData("WordRecognized")
        prob = data[1]
        if prob != prev:
            prev = prob
            print("data: %s" % data[0])
            break
    asr.unsubscribe(ip)
    return data[0]





tts = ALProxy("ALTextToSpeech", "10.0.0.64", 9559)
tts.say("Hello, I am Pepper and I will be your cooking assistant today. How many ingredients do you have?")
number = "zero"

while True:
    number = recognize_1_10()
    say = "I should now take " + number + " picture"
    if number != "one":
        say += "s"
    say += ". Do you confirm this number?"
    tts.say(say)
    conf = recognize_yes_no()
    if conf == "yes":
        break
    else:
        tts.say("I apologise for making a mistake. Could you repeat how many ingredients you have?")

pic = 0
if number == "one":
    pic = 1
elif number == "two":
    pic = 2
elif number == "three":
    pic = 3
elif number == "four":
    pic = 4
elif number == "five":
    pic = 5
elif number == "six":
    pic = 6        
elif number == "seven":
    pic = 7    
elif number == "eight":
    pic = 8
elif number == "nine":
    pic = 9
elif number == "ten":
    pic = 10

for i in range(pic):
    t = True
    while t:
        tts.say("I will now take a picture of the ingredient. Please put it in front of me.")
        time.sleep(10) # wait 10s for the user to get ready with the ingredient
        tts.say("I will now take a picture and show it to you on your computer.")

        call_image(i)
        tts.say("Do you confirm that the ingredient is clearly visible?")
        recognize_yes_no()
        conf = recognize_yes_no()
        if conf == "yes":
            tts.say("Thank you, the picture was saved.")
            break
        tts.say("No worries, we will try again. Please reposition the ingredient so that it will be visible.")

tts.say("All pictures were successfuly taken. Thank you for your cooperation. I will now try to classify the ingredients you showed me.")
