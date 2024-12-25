#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on December 19, 2024, at 10:40
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '0'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

from pylsl import StreamInfo, StreamOutlet

#Set up LabstreamingLayer stream
info = StreamInfo("Markers", "Markers", 1, 0, "int32")
outlet = StreamOutlet(info) #Broadcast the stream
event_ids = {"start_balloon" = 1, "balloon_popped": 2, "balloon_banked": 3}

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'bart_exp'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '0',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1280, 720]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\MODSIM-MR2\\Desktop\\BART_Study\\bart_exp.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units=None,
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 1.0000, 1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = None
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('intro_resp') is None:
        # initialise intro_resp
        intro_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='intro_resp',
        )
    # create speaker 'pop_sound'
    deviceManager.addDevice(
        deviceName='pop_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('bank_resp') is None:
        # initialise bank_resp
        bank_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='bank_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "intro" ---
    intro_txt = visual.TextBox2(
         win, text='Bu oyunda bir balon şişirme yarışmasında olabildiğince fazla para kazanmaya çalışacaksınız. \n\nŞişirdiğiniz her balondan, balondaki hava kadar para kazanacaksınız. Ancak balonu fazla şişirip patlatırsanız o balondan hiç kazanç sağlayamayacaksınız. Balon patladığında sesini duyabileceksiniz. İstediğiniz zaman balonu şişirmeyi bırakıp, paranızı alıp yeni balona geçebilirsiniz.\n\nHer bir balon patlamadan önce pompalayabildiğiniz hava miktarı farklıdır. \n\nLütfen \n     Balonu şişirmek için BOŞLUK tuşuna\n     Balonu şişirmeyi sonlandırıp paranızı almak ve yeni balona geçmek için ENTER tuşuna basın. \n\nLütfen devam etmek için BOŞLUK tuşunu kullanın.\n', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.04,
         size=(1, 1.2), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='intro_txt',
         depth=0, autoLog=True,
    )
    intro_resp = keyboard.Keyboard(deviceName='intro_resp')
    pop_sound = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker='pop_sound',    name='pop_sound'
    )
    pop_sound.setVolume(0.0)
    
    # --- Initialize components for Routine "intro_start" ---
    # Run 'Begin Experiment' code from code_2
    countdown = 5
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "balloon_res" ---
    
    # --- Initialize components for Routine "trial" ---
    bank_resp = keyboard.Keyboard(deviceName='bank_resp')
    # Run 'Begin Experiment' code from update_money
    bankedEarnings=5.0
    balloonEarnings = ''
    bankedText = ''
    lastBalloonEarnings=0.0
    thisBalloonEarnings=0.0
    reminderKeys = visual.TextBox2(
         win, text='Balonu şişirmek için: BOŞLUK tuşu\nYeni balona geçmek için: ENTER tuşu', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(-0.7, -0.7), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.2), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='reminderKeys',
         depth=-2, autoLog=True,
    )
    balloon_value = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(-0.7, 0.7), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.2), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='balloon_value',
         depth=-3, autoLog=True,
    )
    banked_value = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0.7, 0.7), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.2), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='banked_value',
         depth=-4, autoLog=True,
    )
    # Run 'Begin Experiment' code from setBalloonSize
    balloonSize=0.08
    balloonMsgHeight=0.01
    trial_count = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0.7, -0.7), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.2), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='trial_count',
         depth=-6, autoLog=True,
    )
    balloon = visual.ImageStim(
        win=win,
        name='balloon', units='height', 
        image='assets/redBalloon.png', mask=None, anchor='center',
        ori=-90.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    
    # --- Initialize components for Routine "feedback" ---
    # Run 'Begin Experiment' code from check_popped
    feedbackText=""
    feedbackTxt = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.4, 0.2), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='feedbackTxt',
         depth=-1, autoLog=True,
    )
    reminderKeys_2 = visual.TextBox2(
         win, text='Balonu şişirmek için: BOŞLUK tuşu\nYeni balona geçmek için: ENTER tuşu', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(-0.7, -0.7), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.2), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='reminderKeys_2',
         depth=-2, autoLog=True,
    )
    balloon_value_2 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(-0.7, 0.7), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.2), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='balloon_value_2',
         depth=-3, autoLog=True,
    )
    banked_value_2 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0.7, 0.7), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.2), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='banked_value_2',
         depth=-4, autoLog=True,
    )
    trial_count_2 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0.7, -0.7), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.2), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='trial_count_2',
         depth=-5, autoLog=True,
    )
    
    # --- Initialize components for Routine "final_score" ---
    totalScore = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(1, 0.7), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='totalScore',
         depth=-1, autoLog=True,
    )
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "intro" ---
    # create an object to store info about Routine intro
    intro = data.Routine(
        name='intro',
        components=[intro_txt, intro_resp, pop_sound],
    )
    intro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    intro_txt.reset()
    # create starting attributes for intro_resp
    intro_resp.keys = []
    intro_resp.rt = []
    _intro_resp_allKeys = []
    pop_sound.setSound('assets/bang.mp3', secs=1.0, hamming=True)
    pop_sound.setVolume(0.0, log=False)
    pop_sound.seek(0)
    # store start times for intro
    intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro.tStart = globalClock.getTime(format='float')
    intro.status = STARTED
    thisExp.addData('intro.started', intro.tStart)
    intro.maxDuration = None
    # keep track of which components have finished
    introComponents = intro.components
    for thisComponent in intro.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro" ---
    intro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_txt* updates
        
        # if intro_txt is starting this frame...
        if intro_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_txt.frameNStart = frameN  # exact frame index
            intro_txt.tStart = t  # local t and not account for scr refresh
            intro_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_txt.started')
            # update status
            intro_txt.status = STARTED
            intro_txt.setAutoDraw(True)
        
        # if intro_txt is active this frame...
        if intro_txt.status == STARTED:
            # update params
            pass
        
        # *intro_resp* updates
        waitOnFlip = False
        
        # if intro_resp is starting this frame...
        if intro_resp.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            intro_resp.frameNStart = frameN  # exact frame index
            intro_resp.tStart = t  # local t and not account for scr refresh
            intro_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_resp.started')
            # update status
            intro_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_resp.status == STARTED and not waitOnFlip:
            theseKeys = intro_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_resp_allKeys.extend(theseKeys)
            if len(_intro_resp_allKeys):
                intro_resp.keys = _intro_resp_allKeys[-1].name  # just the last key pressed
                intro_resp.rt = _intro_resp_allKeys[-1].rt
                intro_resp.duration = _intro_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *pop_sound* updates
        
        # if pop_sound is starting this frame...
        if pop_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            pop_sound.frameNStart = frameN  # exact frame index
            pop_sound.tStart = t  # local t and not account for scr refresh
            pop_sound.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('pop_sound.started', tThisFlipGlobal)
            # update status
            pop_sound.status = STARTED
            pop_sound.play(when=win)  # sync with win flip
        
        # if pop_sound is stopping this frame...
        if pop_sound.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > pop_sound.tStartRefresh + 1.0-frameTolerance or pop_sound.isFinished:
                # keep track of stop time/frame for later
                pop_sound.tStop = t  # not accounting for scr refresh
                pop_sound.tStopRefresh = tThisFlipGlobal  # on global time
                pop_sound.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pop_sound.stopped')
                # update status
                pop_sound.status = FINISHED
                pop_sound.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[pop_sound]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            intro.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro" ---
    for thisComponent in intro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro
    intro.tStop = globalClock.getTime(format='float')
    intro.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro.stopped', intro.tStop)
    # check responses
    if intro_resp.keys in ['', [], None]:  # No response was made
        intro_resp.keys = None
    thisExp.addData('intro_resp.keys',intro_resp.keys)
    if intro_resp.keys != None:  # we had a response
        thisExp.addData('intro_resp.rt', intro_resp.rt)
        thisExp.addData('intro_resp.duration', intro_resp.duration)
    pop_sound.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    intro_loop = data.TrialHandler2(
        name='intro_loop',
        nReps=5.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(intro_loop)  # add the loop to the experiment
    thisIntro_loop = intro_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisIntro_loop.rgb)
    if thisIntro_loop != None:
        for paramName in thisIntro_loop:
            globals()[paramName] = thisIntro_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisIntro_loop in intro_loop:
        currentLoop = intro_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisIntro_loop.rgb)
        if thisIntro_loop != None:
            for paramName in thisIntro_loop:
                globals()[paramName] = thisIntro_loop[paramName]
        
        # --- Prepare to start Routine "intro_start" ---
        # create an object to store info about Routine intro_start
        intro_start = data.Routine(
            name='intro_start',
            components=[text],
        )
        intro_start.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        countdownText = f"The experiment will start in {countdown} seconds."
        countdown = countdown -1 
        text.setText(countdownText)
        # store start times for intro_start
        intro_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        intro_start.tStart = globalClock.getTime(format='float')
        intro_start.status = STARTED
        thisExp.addData('intro_start.started', intro_start.tStart)
        intro_start.maxDuration = None
        # keep track of which components have finished
        intro_startComponents = intro_start.components
        for thisComponent in intro_start.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "intro_start" ---
        # if trial has changed, end Routine now
        if isinstance(intro_loop, data.TrialHandler2) and thisIntro_loop.thisN != intro_loop.thisTrial.thisN:
            continueRoutine = False
        intro_start.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                intro_start.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in intro_start.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "intro_start" ---
        for thisComponent in intro_start.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for intro_start
        intro_start.tStop = globalClock.getTime(format='float')
        intro_start.tStopRefresh = tThisFlipGlobal
        thisExp.addData('intro_start.stopped', intro_start.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if intro_start.maxDurationReached:
            routineTimer.addTime(-intro_start.maxDuration)
        elif intro_start.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 5.0 repeats of 'intro_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=30.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=1832, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "balloon_res" ---
        # create an object to store info about Routine balloon_res
        balloon_res = data.Routine(
            name='balloon_res',
            components=[],
        )
        balloon_res.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from balloon_res
        balloonSize=0.05
        popped=False
        nPumps=0
        no_array = list(range(1,129))
        # store start times for balloon_res
        balloon_res.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        balloon_res.tStart = globalClock.getTime(format='float')
        balloon_res.status = STARTED
        thisExp.addData('balloon_res.started', balloon_res.tStart)
        balloon_res.maxDuration = None
        # keep track of which components have finished
        balloon_resComponents = balloon_res.components
        for thisComponent in balloon_res.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "balloon_res" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        balloon_res.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                balloon_res.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in balloon_res.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "balloon_res" ---
        for thisComponent in balloon_res.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for balloon_res
        balloon_res.tStop = globalClock.getTime(format='float')
        balloon_res.tStopRefresh = tThisFlipGlobal
        thisExp.addData('balloon_res.stopped', balloon_res.tStop)
        # the Routine "balloon_res" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        pump = data.TrialHandler2(
            name='pump',
            nReps=128.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(pump)  # add the loop to the experiment
        thisPump = pump.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPump.rgb)
        if thisPump != None:
            for paramName in thisPump:
                globals()[paramName] = thisPump[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPump in pump:
            currentLoop = pump
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPump.rgb)
            if thisPump != None:
                for paramName in thisPump:
                    globals()[paramName] = thisPump[paramName]
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[bank_resp, reminderKeys, balloon_value, banked_value, trial_count, balloon],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            #Send LSL Marker when balloon resets
            outlet.push_sample([event_ids["start_balloon"]]) # Push event marker
            # update component parameters for each repeat
            # create starting attributes for bank_resp
            bank_resp.keys = []
            bank_resp.rt = []
            _bank_resp_allKeys = []
            # Run 'Begin Routine' code from update_money
            thisBalloonEarnings=(pump.thisN+1)*0.10
            balloonEarnings = "Bu balondan elde edilen para:\n" + str(round(thisBalloonEarnings, 2)) + " TL"
            bankedText = "Şu ana kadarki toplam kazanç:\n" + str(round(bankedEarnings, 2)) + "TL"
            reminderKeys.reset()
            balloon_value.reset()
            banked_value.reset()
            # Run 'Begin Routine' code from setBalloonSize
            balloon.setPos([0, balloonSize/2-.5])
            balloon.setSize(balloonSize)
            trial_count.reset()
            trial_count.setText('Balon: ' + str(trials.thisN+1) +'/' + str(trials.nTotal))
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            # if trial has changed, end Routine now
            if isinstance(pump, data.TrialHandler2) and thisPump.thisN != pump.thisTrial.thisN:
                continueRoutine = False
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *bank_resp* updates
                waitOnFlip = False
                
                # if bank_resp is starting this frame...
                if bank_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    bank_resp.frameNStart = frameN  # exact frame index
                    bank_resp.tStart = t  # local t and not account for scr refresh
                    bank_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(bank_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bank_resp.started')
                    # update status
                    bank_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(bank_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(bank_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if bank_resp.status == STARTED and not waitOnFlip:
                    theseKeys = bank_resp.getKeys(keyList=["space", "return"], ignoreKeys=["escape"], waitRelease=False)
                    _bank_resp_allKeys.extend(theseKeys)
                    if len(_bank_resp_allKeys):
                        bank_resp.keys = _bank_resp_allKeys[-1].name  # just the last key pressed
                        bank_resp.rt = _bank_resp_allKeys[-1].rt
                        bank_resp.duration = _bank_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *reminderKeys* updates
                
                # if reminderKeys is starting this frame...
                if reminderKeys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    reminderKeys.frameNStart = frameN  # exact frame index
                    reminderKeys.tStart = t  # local t and not account for scr refresh
                    reminderKeys.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(reminderKeys, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'reminderKeys.started')
                    # update status
                    reminderKeys.status = STARTED
                    reminderKeys.setAutoDraw(True)
                
                # if reminderKeys is active this frame...
                if reminderKeys.status == STARTED:
                    # update params
                    pass
                
                # *balloon_value* updates
                
                # if balloon_value is starting this frame...
                if balloon_value.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    balloon_value.frameNStart = frameN  # exact frame index
                    balloon_value.tStart = t  # local t and not account for scr refresh
                    balloon_value.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(balloon_value, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'balloon_value.started')
                    # update status
                    balloon_value.status = STARTED
                    balloon_value.setAutoDraw(True)
                
                # if balloon_value is active this frame...
                if balloon_value.status == STARTED:
                    # update params
                    balloon_value.setText(balloonEarnings, log=False)
                
                # *banked_value* updates
                
                # if banked_value is starting this frame...
                if banked_value.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    banked_value.frameNStart = frameN  # exact frame index
                    banked_value.tStart = t  # local t and not account for scr refresh
                    banked_value.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(banked_value, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'banked_value.started')
                    # update status
                    banked_value.status = STARTED
                    banked_value.setAutoDraw(True)
                
                # if banked_value is active this frame...
                if banked_value.status == STARTED:
                    # update params
                    banked_value.setText(bankedText, log=False)
                # Run 'Each Frame' code from setBalloonSize
                balloonSize=0.1+(pump.thisN+1)*0.015
                
                # *trial_count* updates
                
                # if trial_count is starting this frame...
                if trial_count.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    trial_count.frameNStart = frameN  # exact frame index
                    trial_count.tStart = t  # local t and not account for scr refresh
                    trial_count.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_count, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_count.started')
                    # update status
                    trial_count.status = STARTED
                    trial_count.setAutoDraw(True)
                
                # if trial_count is active this frame...
                if trial_count.status == STARTED:
                    # update params
                    pass
                
                # *balloon* updates
                
                # if balloon is starting this frame...
                if balloon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    balloon.frameNStart = frameN  # exact frame index
                    balloon.tStart = t  # local t and not account for scr refresh
                    balloon.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(balloon, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'balloon.started')
                    # update status
                    balloon.status = STARTED
                    balloon.setAutoDraw(True)
                
                # if balloon is active this frame...
                if balloon.status == STARTED:
                    # update params
                    balloon.setPos((0, balloonSize/2-.5), log=False)
                    balloon.setSize(balloonSize, log=False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # Run 'End Routine' code from update_money
            #calculate cash 'earned'
            chosen = no_array[randint(0,len(no_array))]
            no_array.remove(chosen)
            print(chosen)
            
            if chosen == 1:
                popped = True
            else:
                popped = False
            
            # if balloon popped reset the earnings
            if popped:
              thisBalloonEarnings=0.0
              lastBalloonEarnings=0.0
            else:
                lastBalloonEarnings=thisBalloonEarnings
            
            if 'return' in bank_resp.keys:
                pump.finished = True
                outlet.push_sample([event_ids["balloon_banked"]])# Push event marker
            if popped == True:
                pump.finished = True
                #Send LSL Marker for end of routine
                outlet.push_sample([event_ids["balloon_popped"]]) # Push event marker
              
            # Run 'End Routine' code from setBalloonSize
            #save data
            trials.addData('nPumps', pump.thisN+1)
            trials.addData('size', balloonSize)
            trials.addData('earnings', thisBalloonEarnings)
            trials.addData('popped', popped)
            
            
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 128.0 repeats of 'pump'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[feedbackTxt, reminderKeys_2, balloon_value_2, banked_value_2, trial_count_2],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from check_popped
        # track the banked earnings
        bankedEarnings = bankedEarnings+lastBalloonEarnings
        
        #update the text 
        balloonEarnings = "Bu balondan elde edilen para:\n" + str(round(thisBalloonEarnings, 2)) + " TL"
        bankedText = "Şu ana kadarki toplam kazanç:\n" + str(round(bankedEarnings, 2)) + "TL"
        
        pop_sound.setVolume(1)
        
        # play the pop sound
        if popped==True:
          feedbackText="Balonu patlattınız!"
          pop_sound.play()
        else:
          feedbackText="Bu balondan elde ettiğiniz kazanç:\n" + str(round(lastBalloonEarnings, 2)) + "TL"
          
        feedbackTxt.reset()
        feedbackTxt.setText(feedbackText)
        reminderKeys_2.reset()
        balloon_value_2.reset()
        banked_value_2.reset()
        trial_count_2.reset()
        trial_count_2.setText('Balon: ' + str(trials.thisN+1) +'/' + str(trials.nTotal))
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedbackTxt* updates
            
            # if feedbackTxt is starting this frame...
            if feedbackTxt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedbackTxt.frameNStart = frameN  # exact frame index
                feedbackTxt.tStart = t  # local t and not account for scr refresh
                feedbackTxt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedbackTxt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedbackTxt.started')
                # update status
                feedbackTxt.status = STARTED
                feedbackTxt.setAutoDraw(True)
            
            # if feedbackTxt is active this frame...
            if feedbackTxt.status == STARTED:
                # update params
                pass
            
            # if feedbackTxt is stopping this frame...
            if feedbackTxt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedbackTxt.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    feedbackTxt.tStop = t  # not accounting for scr refresh
                    feedbackTxt.tStopRefresh = tThisFlipGlobal  # on global time
                    feedbackTxt.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedbackTxt.stopped')
                    # update status
                    feedbackTxt.status = FINISHED
                    feedbackTxt.setAutoDraw(False)
            
            # *reminderKeys_2* updates
            
            # if reminderKeys_2 is starting this frame...
            if reminderKeys_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reminderKeys_2.frameNStart = frameN  # exact frame index
                reminderKeys_2.tStart = t  # local t and not account for scr refresh
                reminderKeys_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reminderKeys_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reminderKeys_2.started')
                # update status
                reminderKeys_2.status = STARTED
                reminderKeys_2.setAutoDraw(True)
            
            # if reminderKeys_2 is active this frame...
            if reminderKeys_2.status == STARTED:
                # update params
                pass
            
            # if reminderKeys_2 is stopping this frame...
            if reminderKeys_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > reminderKeys_2.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    reminderKeys_2.tStop = t  # not accounting for scr refresh
                    reminderKeys_2.tStopRefresh = tThisFlipGlobal  # on global time
                    reminderKeys_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'reminderKeys_2.stopped')
                    # update status
                    reminderKeys_2.status = FINISHED
                    reminderKeys_2.setAutoDraw(False)
            
            # *balloon_value_2* updates
            
            # if balloon_value_2 is starting this frame...
            if balloon_value_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                balloon_value_2.frameNStart = frameN  # exact frame index
                balloon_value_2.tStart = t  # local t and not account for scr refresh
                balloon_value_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(balloon_value_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'balloon_value_2.started')
                # update status
                balloon_value_2.status = STARTED
                balloon_value_2.setAutoDraw(True)
            
            # if balloon_value_2 is active this frame...
            if balloon_value_2.status == STARTED:
                # update params
                balloon_value_2.setText(balloonEarnings, log=False)
            
            # if balloon_value_2 is stopping this frame...
            if balloon_value_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > balloon_value_2.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    balloon_value_2.tStop = t  # not accounting for scr refresh
                    balloon_value_2.tStopRefresh = tThisFlipGlobal  # on global time
                    balloon_value_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'balloon_value_2.stopped')
                    # update status
                    balloon_value_2.status = FINISHED
                    balloon_value_2.setAutoDraw(False)
            
            # *banked_value_2* updates
            
            # if banked_value_2 is starting this frame...
            if banked_value_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                banked_value_2.frameNStart = frameN  # exact frame index
                banked_value_2.tStart = t  # local t and not account for scr refresh
                banked_value_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(banked_value_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'banked_value_2.started')
                # update status
                banked_value_2.status = STARTED
                banked_value_2.setAutoDraw(True)
            
            # if banked_value_2 is active this frame...
            if banked_value_2.status == STARTED:
                # update params
                banked_value_2.setText(bankedText, log=False)
            
            # if banked_value_2 is stopping this frame...
            if banked_value_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > banked_value_2.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    banked_value_2.tStop = t  # not accounting for scr refresh
                    banked_value_2.tStopRefresh = tThisFlipGlobal  # on global time
                    banked_value_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'banked_value_2.stopped')
                    # update status
                    banked_value_2.status = FINISHED
                    banked_value_2.setAutoDraw(False)
            
            # *trial_count_2* updates
            
            # if trial_count_2 is starting this frame...
            if trial_count_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_count_2.frameNStart = frameN  # exact frame index
                trial_count_2.tStart = t  # local t and not account for scr refresh
                trial_count_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_count_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_count_2.started')
                # update status
                trial_count_2.status = STARTED
                trial_count_2.setAutoDraw(True)
            
            # if trial_count_2 is active this frame...
            if trial_count_2.status == STARTED:
                # update params
                pass
            
            # if trial_count_2 is stopping this frame...
            if trial_count_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trial_count_2.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    trial_count_2.tStop = t  # not accounting for scr refresh
                    trial_count_2.tStopRefresh = tThisFlipGlobal  # on global time
                    trial_count_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_count_2.stopped')
                    # update status
                    trial_count_2.status = FINISHED
                    trial_count_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback.maxDurationReached:
            routineTimer.addTime(-feedback.maxDuration)
        elif feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        thisExp.nextEntry()
        
    # completed 30.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "final_score" ---
    # create an object to store info about Routine final_score
    final_score = data.Routine(
        name='final_score',
        components=[totalScore],
    )
    final_score.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code
    scoreText="Tebrikler! Toplamda \n" + str(round(bankedEarnings, 2)) + "TL kazandınız!"
    totalScore.reset()
    totalScore.setText(scoreText)
    # store start times for final_score
    final_score.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    final_score.tStart = globalClock.getTime(format='float')
    final_score.status = STARTED
    thisExp.addData('final_score.started', final_score.tStart)
    final_score.maxDuration = None
    # keep track of which components have finished
    final_scoreComponents = final_score.components
    for thisComponent in final_score.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "final_score" ---
    final_score.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *totalScore* updates
        
        # if totalScore is starting this frame...
        if totalScore.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            totalScore.frameNStart = frameN  # exact frame index
            totalScore.tStart = t  # local t and not account for scr refresh
            totalScore.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(totalScore, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'totalScore.started')
            # update status
            totalScore.status = STARTED
            totalScore.setAutoDraw(True)
        
        # if totalScore is active this frame...
        if totalScore.status == STARTED:
            # update params
            pass
        
        # if totalScore is stopping this frame...
        if totalScore.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > totalScore.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                totalScore.tStop = t  # not accounting for scr refresh
                totalScore.tStopRefresh = tThisFlipGlobal  # on global time
                totalScore.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'totalScore.stopped')
                # update status
                totalScore.status = FINISHED
                totalScore.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            final_score.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in final_score.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "final_score" ---
    for thisComponent in final_score.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for final_score
    final_score.tStop = globalClock.getTime(format='float')
    final_score.tStopRefresh = tThisFlipGlobal
    thisExp.addData('final_score.stopped', final_score.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if final_score.maxDurationReached:
        routineTimer.addTime(-final_score.maxDuration)
    elif final_score.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
