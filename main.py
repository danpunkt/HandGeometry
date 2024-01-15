import cv2
import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.signal import find_peaks

import HandClasses as hc
font                   = cv2.FONT_HERSHEY_PLAIN
lineType               = 2


# Infos:
# Das Programm läuft mit den aktuellen Parametern über die 6 Beispielbilder und Vergleich die dort gefunden Merkmale
# mit dem ausgewählten Merkmalssatz.
# Alle 3 Sekunden wird ein neues Bild geladen. Druck einer beliebigen Taste spingt zum nächsten Bild.
# Druck der Taste 'q' beendet das Programm.


# ___________________________________Parameter________________________________________________

debugPics = 0      # (Wert 1) Zeigt Debugbilder während der Verarbeitung an.

live = 0           # Wählt aus ob im Live Modus (1) oder mit den gespeicherten Bilder simuliert werden soll (0)

iKamera = 1        # Wählt die Kamera aus. Relevant falls das Programm im Live Modus läuft

Verifizieren = 2   # (Werte 1-6). Wählt die Person/den Merkmalssatz aus, mit dem die Bilder verglichen werden

threshold = 80     # Der gefundene Grenzwert für eine erfolgreiche Verifizierung



# Eingelernte Merkmalssätze aus Testbildern
TeachedPers1 = [41.012193308819754, 128.7245120402482, 222, 42.05948168962618, 152.0855022676389, 301, 48.25971404805462, 170.2292571798397, 350, 61.98386886924694, 171.04677722775136, 359, 86.37708029332781, 148.49242404917499, 275]
TeachedPers2 = [42.44997055358225, 126.05157674539419, 239, 40.70626487409524, 158.75767697972907, 317, 44.181444068749045, 170.98830369355676, 355, 50.44799302251776, 163.22989922192565, 343, 85.38149682454625, 151.32745950421557, 306]
TeachedPers3 = [49.4064773081425, 132.41223508422476, 243, 47.41307836451879, 159.48040632002414, 327, 49.24428900898052, 175.524927004685, 363, 58.42088667591412, 167.29913329123974, 353, 90.82400563727631, 147.75655653811103, 281]
TeachedPers4 = [44.77722635447622, 126.0634760745554, 217, 36.796738985948195, 152.11837495845134, 286, 40.496913462633174, 169.00887550658396, 334, 54.3415126767741, 162.00308639035245, 342, 79.88116173416608, 139.20129309744217, 282]
TeachedPers5 = [46.22769732530488, 130.09611831257686, 232, 43.01162633521314, 156.58863304850706, 303, 46.389654018972806, 176.91806012954132, 370, 62.76941930590086, 162.2374802565671, 352, 87.64131445842195, 142.3411395205195, 286]
TeachedPers6 = [60.8276253029822, 156.348968656656, 280, 54.08326913195984, 179.2344832893492, 350, 49.01020301937138, 194.4350791395421, 402, 71.34423592694787, 186.72439583514523, 377, 84.11896337925236, 151.18200951171406, 286]

TeachedPersons = [TeachedPers1, TeachedPers2, TeachedPers3, TeachedPers4, TeachedPers5, TeachedPers6]
TeachedPers = TeachedPersons[Verifizieren-1]

# Testbilder zum Vergleich mit den Merkmalssätzen
TestBilder = ['Hand_Testdata/Pers1.png', 'Hand_Testdata/Pers2.png', 'Hand_Testdata/Pers3.png', 'Hand_Testdata/Pers4.png', 'Hand_Testdata/Pers5.png', 'Hand_Testdata/Pers6.png']

#setze Kameraeinstellungen für den Live Modus
if live == 1:
    print("Versuche Zugriff auf Kamera " + str(iKamera))
    cap = cv2.VideoCapture(iKamera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)
    #cap.set(cv2.CAP_PROP_EXPOSURE, 0)
    print("live mode")


# Findet den nächsten Punkt auf der Kontur
def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index], closest_index

# Fügt Punkte zu den bereits gefundenen hinzu
def addVecPoints(self, other, start):
    newx = start[0] + (self[0] - other[0])
    newy = start[1] + (self[1] - other[1])
    return int(newx), int(newy)

# Erstellt den Vektor für die Verifizierung
def genValidationVec(FingerVec, ValidationVec,Bild,Kontur):
    dDistLR = distance.euclidean(FingerVec[1][0], FingerVec[2][0])
    MFLx = FingerVec[1][0][0]
    MFLy = FingerVec[1][0][1]
    MFRx = FingerVec[2][0][0]
    MFRy = FingerVec[2][0][1]
    MidMF = (int((MFLx+MFRx)/2), int((MFLy+MFRy)/2))
    dDistTM = distance.euclidean(MidMF, FingerVec[0][0])
    circCont = FingerVec[1][1] - FingerVec[2][1]
    ValidationVec.append(dDistLR)
    ValidationVec.append(dDistTM)
    ValidationVec.append(circCont)
    # Einzechnen der Features
    cv2.circle(Bild, FingerVec[0][0], 2, (0, 255, 0), -1)
    cv2.circle(Bild, MidMF, 2, (0, 255, 0), -1)
    cv2.line(Bild, FingerVec[0][0], MidMF, (0, 255, 0), 1)

    cv2.circle(Bild, FingerVec[1][0], 2, (0, 255, 0), -1)
    cv2.circle(Bild, FingerVec[2][0], 2, (0, 255, 0), -1)
    cv2.line(Bild, FingerVec[1][0], FingerVec[2][0], (0, 255, 0), 1)

    cv2.drawContours(Bild,Kontur[FingerVec[2][1]:FingerVec[1][1]],-1,(0,255,0),1)
    plt.imshow(Bild)
    return Bild, ValidationVec



# Vordefinierte Variablen
iTest = 0
vec_KleinerFinger = []
vec_RingFinger = []
vec_MittelFinger = []
vec_ZeigeFinger = []
vec_Daumen = []

FlipIMG = True
GoLive = True
vec_DistancesFinger = []

# Porgrammschleife
while GoLive:
    if live == 1:
        iSleep = 50
        _, IMG_RGB = cap.read()

    else:
        iSleep = 3000
        IMG_RGB = cv2.imread(TestBilder[iTest], 1)
        if iTest < (len(TestBilder) - 1):
            iTest = iTest +1
        else:
            iTest = 0
        print('TestBild: ' + str(iTest))


    if FlipIMG:
        IMG_RGB_Flip = cv2.flip(IMG_RGB, 0)

    if debugPics: cv2.imshow('debug', IMG_RGB) #Debug Bild

    # Zurückgesetzte Variablen
    vec_DistancesFinger = []
    vec_DistancesSearchThumb = []
    vec_KleinerFinger = []
    vec_RingFinger = []
    vec_MittelFinger = []
    vec_ZeigeFinger = []
    vec_Daumen = []
    ERR = 0
    vec_TestDrawPoint = []
    vec_ValidationVec = []
    IMG_Gray = cv2.cvtColor(IMG_RGB_Flip, cv2.COLOR_BGR2GRAY)

    IMG_mask1 = np.zeros((IMG_Gray.shape[0], IMG_Gray.shape[1]), dtype="uint8")
    IMG_mask2 = np.zeros((IMG_Gray.shape[0], IMG_Gray.shape[1]), dtype="uint8")

    if debugPics: cv2.imshow('debug', IMG_Gray)  # Debug Bild

    # ------------------------------------
    # Erweiterte Binarisierung um Lücken zu schließen und nur die Hand zu erkennen

    _, IMG_Binary = cv2.threshold(IMG_Gray, 75, 255, cv2.THRESH_BINARY)

    if debugPics: cv2.imshow('debuga', IMG_Binary)  # Debug Bild

    kernel = np.ones((15, 15), np.uint8)
    IMG_Binary = cv2.morphologyEx(IMG_Binary, cv2.MORPH_CLOSE, kernel)
    IMG_Binary = cv2.dilate(IMG_Binary, kernel,iterations=2)
    IMG_Gray = cv2.bitwise_and(IMG_Gray, IMG_Binary)
    if debugPics: cv2.imshow('debugb', IMG_Gray)  # Debug Bild

    _, IMG_Binary = cv2.threshold(IMG_Gray, 35, 255, cv2.THRESH_BINARY)

    if debugPics: cv2.imshow('debugc', IMG_Binary)  # Debug Bild

    # ------------------------------------
    # Suche Nach einem "annähernd" Mittleren Punkt auf der Handfläche

    cnts_FindAreas, _ = cv2.findContours(IMG_Binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # Kontur der Hand
    try:
        cnt_FindArea = max(cnts_FindAreas, key=len)
    except ValueError:
        continue
    area = cv2.contourArea(cnt_FindArea)

    _, ppoint, max_ddd, _ = hc.mark_hand_center(IMG_Binary, cnt_FindArea, 0.008) # Suche den Mittelpunkt
    cv2.circle(IMG_RGB_Flip, ppoint, 2, (255, 0, 0), 2)

    if debugPics: cv2.imshow('debug2', IMG_RGB_Flip)  # Debug Bild

    cv2.rectangle(IMG_Binary,(0,ppoint[1]-100),(IMG_Gray.shape[1],0),0,-1) # Trenne die Hand vom Handgelenk

    if debugPics: cv2.imshow('debug', IMG_Binary)  # Debug Bild

    # ------------------------------------
    # Suche die "richtige" Kontur der Hand nach den Vorverarbeitungsprozessen

    cnts_Hand, hierarchy_Hand = cv2.findContours(IMG_Binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    try:
        cnt_Hand = max(cnts_Hand, key=len)
    except ValueError:
        continue

    cntSQ_Hand = np.vstack(cnt_Hand).squeeze()

    # ------------------------------------
    # Polarabstandsprojektion

    # Aus der Kontur werden die relevanten Punkte / Fingerspitzen und Fingertäler gewonnen
    # Gefunden mit zwei Druchläufen vom Mittelpunkt & vom Startpunkt der Kontur
    # Die Beizeichnungen Finger und Thumb sind noch Historisch bei der Programmierung entstanden
    # In beiden Fällen werden alle Finger gesucht. Die Bezeichnungen werden jetzt nicht mehr angepasst.

    startPoint = (cntSQ_Hand[0][0], cntSQ_Hand[0][1])
    cv2.circle(IMG_mask2, startPoint, 10, 255, 1)
    cv2.drawContours(IMG_mask2, cnt_Hand, -1, 255, 1)

    if debugPics: cv2.imshow('debug2', IMG_mask2)  # Debug Bild

    # Polarabstandsprojektion mit zwei Ausgangspunkten
    for i in range(len(cntSQ_Hand)):
        Cnt_Point = (cntSQ_Hand[i][0],cntSQ_Hand[i][1])
        dFindFingers = distance.euclidean(Cnt_Point, ppoint)
        dFindThumb = distance.euclidean(Cnt_Point, startPoint)
        vec_DistancesFinger.append(dFindFingers)
        vec_DistancesSearchThumb.append(dFindThumb)

    # Maxima / Minima in den Werten finden die vom Startpunkt der Kontur gefunden wurden (mit eigener Filterlänge)
    iFiltLength = 4
    filtFunc = np.convolve(vec_DistancesSearchThumb, np.full(iFiltLength, 1 / iFiltLength), 'same')
    maxFromStart, _ = find_peaks(filtFunc, distance=250)
    minFromStart, _ = find_peaks(-filtFunc, distance=250)
    #np.savetxt('fingers1.txt', filtFunc)

    # Maxima / Minima in den Werten finden die vom Mittelpunkt der Handgefunden wurden (mit eigener Filterlänge)
    iFiltLength = 160#iWert
    filtFunc = np.convolve(vec_DistancesFinger, np.full(iFiltLength, 1 / iFiltLength), 'same')
    maxFromMid, _ = find_peaks(filtFunc, distance=250)
    minFromMid, _ = find_peaks(-filtFunc, distance=50)
    #np.savetxt('fingers2.txt', filtFunc)


    #Backupwerte falls nichts gefunden wird (livebild)
    if not minFromMid.size > 0:
        minFromMid = [0]
    if not maxFromMid.size > 0:
        maxFromMid = [0,1]

    # Einzeichnen in das Bild
    for i in range(len(maxFromStart)):
        ptDrawPoint = (cntSQ_Hand[maxFromStart[i]][0], cntSQ_Hand[maxFromStart[i]][1])
        cv2.circle(IMG_mask2, ptDrawPoint, 5, 125, 2)

    if debugPics: cv2.imshow('debug3', IMG_mask2)  # Debug Bild

    for i in range(len(minFromStart)):
        ptDrawPoint = (cntSQ_Hand[minFromStart[i]][0], cntSQ_Hand[minFromStart[i]][1])
        cv2.circle(IMG_mask2, ptDrawPoint, 5, 125, 2)

    if debugPics: cv2.imshow('debug3', IMG_mask2)  # Debug Bild

    for i in range(len(maxFromMid)):
        ptDrawPoint = (cntSQ_Hand[maxFromMid[i]][0], cntSQ_Hand[maxFromMid[i]][1])
        cv2.circle(IMG_mask2, ptDrawPoint, 5, 255, 2)

    if debugPics: cv2.imshow('debug3', IMG_mask2)  # Debug Bild

    for i in range(len(minFromMid)):
        ptDrawPoint = (cntSQ_Hand[minFromMid[i]][0], cntSQ_Hand[minFromMid[i]][1])
        cv2.circle(IMG_mask2, ptDrawPoint, 5, 255, 2)

    if debugPics: cv2.imshow('debug3', IMG_mask2)  # Debug Bild

    # ------------------------------------
    # Fehlenden Punkte mit Extrapolation finden

    vec_MissingPoints = []
    x = cntSQ_Hand[minFromMid[0:3]][:, 0]
    y = cntSQ_Hand[minFromMid[0:3]][:, 1]
    try:
        f = interp1d(x, y, 'quadratic', fill_value='extrapolate')
    except ValueError:
        continue


    # Suche Punkt an der Außenseite des kleinen Fingers
    predX = cntSQ_Hand[minFromMid[0]][0] - (cntSQ_Hand[minFromMid[1]][0] - cntSQ_Hand[minFromMid[0]][0])
    MissingPoint, MissingIndex = closest_node([predX,f(predX)], cntSQ_Hand) # Suche mit der "Prediction" und der Kurve den nächsten Punkte auf der Kontur
    vec_MissingPoints.append([tuple(MissingPoint),MissingIndex])
    cv2.circle(IMG_mask2, vec_MissingPoints[0][0], 5, 255, -1)

    if debugPics: cv2.imshow('debug4', IMG_mask2)  # Debug Bild

    # Suche Punkt an der Außenseite des Zeigefingers
    predX = cntSQ_Hand[minFromMid[2]][0] + (cntSQ_Hand[minFromMid[2]][0] - cntSQ_Hand[minFromMid[1]][0])
    MissingPoint, MissingIndex = closest_node([predX, f(predX)], cntSQ_Hand) # Suche mit der "Prediction" und der Kurve den nächsten Punkte auf der Kontur
    vec_MissingPoints.append([tuple(MissingPoint),MissingIndex])
    cv2.circle(IMG_mask2, vec_MissingPoints[1][0], 5, 255, -1)


    if debugPics: plt.imshow(IMG_mask2)

    # Suche Punkt an der Außenseite des Daumens - Hier wird nur eine Gerade durch zwei Punkte gelegt
    predXY = addVecPoints(cntSQ_Hand[minFromStart[-1]], list(vec_MissingPoints[-1][0]), cntSQ_Hand[minFromStart[-1]])
    MissingPoint, MissingIndex = closest_node(predXY, cntSQ_Hand)
    vec_MissingPoints.append([tuple(MissingPoint), MissingIndex])
    cv2.circle(IMG_mask2, vec_MissingPoints[2][0], 5, 255, -1)

    cv2.imshow('IMG_mask2', IMG_mask2)

    # Sicherheitsüberprüfung
    if len(minFromStart) < 4 or len(maxFromStart) < 5 or len(minFromMid) < 4 or len(maxFromMid) < 4:
        continue

    # ------------------------------------
    # Merkmale werden generiert und gesammelt

    # Kleiner Finger
    TmpPointTopIndex = int((maxFromStart[0] + maxFromMid[0])/2)
    TmpPointTop = [(cntSQ_Hand[TmpPointTopIndex][0], cntSQ_Hand[TmpPointTopIndex][1]), TmpPointTopIndex]
    vec_KleinerFinger.append(TmpPointTop)
    TempPointMinLinksIndex = minFromStart[0]
    TempPointMinLinks = [(cntSQ_Hand[minFromStart[0]][0], cntSQ_Hand[minFromStart[0]][1]), TempPointMinLinksIndex]
    vec_KleinerFinger.append(TempPointMinLinks)
    TempPointMinRechts = vec_MissingPoints[0]
    vec_KleinerFinger.append(TempPointMinRechts)
    # Ringfinger
    TmpPointTopIndex = int((maxFromStart[1] + maxFromMid[1])/2)
    TmpPointTop = [(cntSQ_Hand[TmpPointTopIndex][0], cntSQ_Hand[TmpPointTopIndex][1]), TmpPointTopIndex]
    vec_RingFinger.append(TmpPointTop)
    TempPointMinLinksIndex = minFromStart[1]
    TempPointMinLinks = [(cntSQ_Hand[TempPointMinLinksIndex][0], cntSQ_Hand[TempPointMinLinksIndex][1]), TempPointMinLinksIndex]
    vec_RingFinger.append(TempPointMinLinks)
    TempPointMinRechtsIndex = minFromMid[0]
    TempPointMinRechts = [(cntSQ_Hand[TempPointMinRechtsIndex][0], cntSQ_Hand[TempPointMinRechtsIndex][1]), TempPointMinRechtsIndex]
    vec_RingFinger.append(TempPointMinRechts)
    # Mittelfinger
    TmpPointTopIndex = int((maxFromStart[2] + maxFromMid[2])/2)
    TmpPointTop = [(cntSQ_Hand[TmpPointTopIndex][0], cntSQ_Hand[TmpPointTopIndex][1]),TmpPointTopIndex]
    vec_MittelFinger.append(TmpPointTop)
    TempPointMinLinksIndex = minFromStart[2]
    TempPointMinLinks = [(cntSQ_Hand[TempPointMinLinksIndex][0], cntSQ_Hand[TempPointMinLinksIndex][1]), TempPointMinLinksIndex]
    vec_MittelFinger.append(TempPointMinLinks)
    TempPointMinRechtsIndex = minFromMid[1]
    TempPointMinRechts = [(cntSQ_Hand[TempPointMinRechtsIndex][0], cntSQ_Hand[TempPointMinRechtsIndex][1]), TempPointMinRechtsIndex]
    vec_MittelFinger.append(TempPointMinRechts)
    # Zeigefinger
    TmpPointTopIndex = int((maxFromStart[3] + maxFromMid[3])/2)
    TmpPointTop = [(cntSQ_Hand[TmpPointTopIndex][0], cntSQ_Hand[TmpPointTopIndex][1]),TmpPointTopIndex]
    vec_ZeigeFinger.append(TmpPointTop)
    TempPointMinLinks = vec_MissingPoints[1]
    vec_ZeigeFinger.append(TempPointMinLinks)
    TempPointMinRechtsIndex = minFromMid[2]
    TempPointMinRechts = [(cntSQ_Hand[TempPointMinRechtsIndex][0], cntSQ_Hand[TempPointMinRechtsIndex][1]), TempPointMinRechtsIndex]
    vec_ZeigeFinger.append(TempPointMinRechts)

    try:
        TmpPointTopIndex = maxFromStart[4]
    except ValueError:
        continue

    # Daumen
    TmpPointTop = [(cntSQ_Hand[TmpPointTopIndex][0], cntSQ_Hand[TmpPointTopIndex][1]), TmpPointTopIndex]
    vec_Daumen.append(TmpPointTop)
    TempPointMinLinks = vec_MissingPoints[2]
    vec_Daumen.append(TempPointMinLinks)
    TempPointMinRechtsIndex = minFromStart[3]
    TempPointMinRechts = [(cntSQ_Hand[TempPointMinRechtsIndex][0], cntSQ_Hand[TempPointMinRechtsIndex][1]),TempPointMinRechtsIndex]
    vec_Daumen.append(TempPointMinRechts)

    # ------------------------------------
    # Merkmalesvektor wird erstellt
    IMG_RGB_Flip, vec_ValidationVec = genValidationVec(vec_KleinerFinger, vec_ValidationVec, IMG_RGB_Flip, cnt_Hand)
    IMG_RGB_Flip, vec_ValidationVec = genValidationVec(vec_RingFinger, vec_ValidationVec, IMG_RGB_Flip, cnt_Hand)
    IMG_RGB_Flip, vec_ValidationVec = genValidationVec(vec_MittelFinger, vec_ValidationVec, IMG_RGB_Flip, cnt_Hand)
    IMG_RGB_Flip, vec_ValidationVec = genValidationVec(vec_ZeigeFinger, vec_ValidationVec, IMG_RGB_Flip, cnt_Hand)
    IMG_RGB_Flip, vec_ValidationVec = genValidationVec(vec_Daumen, vec_ValidationVec, IMG_RGB_Flip, cnt_Hand)



    # ------------------------------------
    # Absolute Distanz
    for i in range(len(vec_ValidationVec)):
        ERR = ERR + abs(vec_ValidationVec[i] - TeachedPers[i])

    print('________________')
    print('ERROR:' + str(ERR))
    print('________________')


    # ------------------------------------
    # Verifikation & Anzeige

    if ERR > threshold:
        sAccess = 'NOT VALID'
        sAccess_fontColor = (0, 0, 255)
        sERRValue = str(format(ERR, '.2f'))
    else:
        sAccess = 'VALID'
        sAccess_fontColor = (0, 255, 0)
        sERRValue = str(format(ERR, '.2f'))

    IMG_RGB = cv2.flip(IMG_RGB_Flip, 0)
    if live == 1:
        IMG_mask1 = cv2.imread('MASKE.png', 1)

        cv2.putText(IMG_RGB, sAccess, (10, 30), font, 2, sAccess_fontColor, lineType)
        cv2.putText(IMG_RGB, sERRValue, (10, 50), font, 1, (255, 255, 255), lineType)
        IMG_RGB = cv2.bitwise_or(IMG_RGB, IMG_mask1)
        cv2.imshow('IMG_RGB', IMG_RGB)
    else:
        cv2.putText(IMG_RGB, sAccess, (10, 30), font, 2, sAccess_fontColor, lineType)
        cv2.putText(IMG_RGB, sERRValue, (10, 50), font, 1, (255, 255, 255), lineType)
        cv2.imshow('IMG_RGB', IMG_RGB)

    IMG_mask2 = cv2.flip(IMG_mask2, 0)
    cv2.imshow('IMG_mask2', IMG_mask2)

    # Abfrage Benutzereingabe
    UIn_Key = cv2.waitKey(iSleep) & 0xff
    if UIn_Key == 113:  # Das ist ein q
        break

# cv2.imwrite('IMG_Gray.png',IMG_Gray)
#UIn_Key = cv2.waitKey(0) & 0xff
if live: cap.release()
cv2.destroyAllWindows()
