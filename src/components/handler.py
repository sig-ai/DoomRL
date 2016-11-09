import ctypes
from sys import exc_info
import traceback
import code

def updateVar(var_name, new_val, frame, local=True):
  if(local):
    frame.f_locals.update({var_name:new_val})
  else:
    frame.f_globals.update({var_name:new_val})
  ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))

def signal_handler(signal, frame):
    rerun = True
    exitFlag = False
    print ""
    print "Entered interrupt handler!"
    compFrame = frame

    #enable getting frame for:
    #  agent
    #  components
    #  baseline
    #return 

    while "general.py" not in compFrame.f_code.co_filename:
        print compFrame.f_code.co_filename
        compFrame = compFrame.f_back
    d={'_frame':compFrame}         
    d.update(compFrame.f_globals)  
    d.update(compFrame.f_locals)

    i = code.InteractiveConsole(d)
    message  = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)
    return

    while True:
      try:
        newCommand = raw_input("(InterruptHandler)>")
        if newCommand == "exit":
          exitFlag = True
          break
        elif newCommand == "resume":
          break
        elif newCommand == "list":
          print __name__, dir()
          print " Name      | Value"
          print "+++++++++++|++++++++++++++++++++"
          for d in dir():
            print " "+d[:10]+" "*(10-len(d[:10]))+"|"+str(eval(d))
        else:
          exec(newCommand)# in compFrame.f_globals, compFrame.f_locals
      except KeyboardInterrupt:
        break
      except:
        e=exc_info()[0]
        print e
        traceback.print_tb(exc_info()[2])
        print "Caught error, continue? (y/n)?"
        response = raw_input()
        print "response "+response 
        if response[0] in ["Y", "y"]:
          continue
        exit(1)
    if exitFlag:
      print "Exiting program"
      updateVar("quit", True, compFrame)
      #exit(1)
      #break
    else:
      print "Resuming execution"
