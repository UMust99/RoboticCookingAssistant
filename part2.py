from naoqi import ALProxy
import math
import motion
from PIL import Image
import qi
import argparse
import sys
import time
import random

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



def retrieve_recipe(ingredients):
    lines = []


    with open('recipes.txt') as f: # open database with recipes
        lines = f.readlines()


    showNext = False
    newRecipe = True
    firstRecipe = []
    oldRecipe = ''
    for line in lines:
        if line[0] == '#':
            showNext = False
            newRecipe = True
            if(oldRecipe != ''):
                firstRecipe.append(oldRecipe)
                oldRecipe = ''
            continue
        
        if(newRecipe == True):
            newRecipe = False
            line = line.strip() #to remove /n
            tmp = line.split(', ') # get the list of all the ingredients
            #print(tmp)
            canDo = True
            for ing in tmp:
                if(ing in ingredients):
                    continue
                canDo = False
                #print(ing)
            if(canDo):
                showNext = True
            continue
        if(showNext == True):
            oldRecipe += ' ' + line
                
    random.shuffle(firstRecipe)
    if len(firstRecipe) == 0:
        return ''
    return(firstRecipe[0])



tts = ALProxy("ALTextToSpeech", "10.0.0.64", 9559)
ingredients = []
with open('ingredients.txt') as f:
    lines = f.readlines()
    ingredients = lines[0].split(' ')
recognized = ''

for i in ingredients:
    if i != '':
        recognized = recognized + i + ' '

tts.say("I recognized " + recognized + ". I will now try to find a good recipe so you will have a delicious meal.")
recipe = retrieve_recipe(ingredients)
if recipe == '':
    tts.say("I am sorry, you don't have enough ingredients to make any of the recipes in my database. Please try again with different ingredients.")

else:
    tts.say("With the available ingredients, I suggest the following recipe." + recipe)
    while True:
            tts.say("Should I repeat the recipe?");
            conf = recognize_yes_no()
            if conf == "no":
                break
            tts.say(recipe)
            time.sleep(10)
    tts.say("Enjoy your meal!")
