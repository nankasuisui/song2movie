import eel,sys

@eel.expose
def close():
    sys.exit()

eel.init("web")
eel.start("index.html",block=False)

while True:
    eel.sleep(1.0)
