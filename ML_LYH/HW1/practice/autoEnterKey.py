#-*-utf-8-*-
import pyautogui
import pyperclip
import datetime
import time

def paste(foo):
	pyperclip.copy(foo)
	pyautogui.hotkey('ctrl', 'v')
def saySth():
	pyautogui.click(625,600)
	foo = u'''当你收到这段文字我估计还在睡觉，咳咳,没错正如你看到的，在我即将成功的时候，我失去了对你电脑的控制。'''
	paste(foo)
	pyautogui.typewrite(["enter"],interval=0.25)
	time.sleep(60)

def main(h=0, m=0):
   while True:
        while True:
            now = datetime.datetime.now()
            print(now)
            if now.hour==7 and now.minute==20:
                break
            time.sleep(20)
        saySth()
#main()