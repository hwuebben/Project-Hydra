import numpy as np

class featureExtraction:

    def __init__(self, pic):
        self.distributionX = np.zeros(np.shape(pic)[0])
        self.distributionY = np.zeros(np.shape(pic)[0])

        for i in range(np.shape(pic)[0]):
            self.distributionX[i] = np.sum(pic[i,:])
            self.distributionY[i] = np.sum(pic[:, i])



    def calcVar(self):

        distributionX = self.distributionX[np.nonzero(self.distributionX)]
        distributionX = np.divide(distributionX,np.max(distributionX))
        distributionY = self.distributionY[np.nonzero(self.distributionY)]
        distributionY = np.divide(distributionY,np.max(distributionY))

        return [np.var(distributionX), np.var(distributionY)]
    """
    berechne die relativen POsitionen der minima und maxima von den
    distribtionen bezogen auf das kleinste Rechteck
    """
    def calcMinMax(self):
        cbl, cbr, cul, cur = self.calcRectangle()
        #print("a: ",cbl, cbr, cul, cur)
        argMaxX = np.argmax(self.distributionX)
        argMaxY = np.argmax(self.distributionY)
        #print("b: ", argMaxX,argMaxY)
        #relative position der maxima:
        rpMaxX = (argMaxX - cbl) / cbr
        rpMaxY = (argMaxY - cul) / cur

        argMinX = np.argmin(self.distributionX)
        argMinY = np.argmin(self.distributionY)
        #print("c: ", argMinX, argMinY)
        #relative position der minima:
        rpMinX = (argMinX - cbl) / cbr
        rpMinY = (argMinY - cul) / cur

        return [rpMaxX,rpMaxY,rpMinX,rpMinY]

    def normalizedDists(self):

        distributionX = self.distributionX[np.nonzero(self.distributionX)]
        distributionX = np.divide(distributionX,np.max(distributionX))


        sumX = np.sum(distributionX)
        lenX = len(distributionX)
        nrParts = 5
        startIndex = 0
        stopIndex = int(lenX/nrParts)
        nrAdditional = lenX % nrParts
        normDistsX = []
        for i in range(nrParts):
            if nrAdditional > 0:
                stopIndex += 1
                nrAdditional -= 1
            partSum = np.sum(distributionX[startIndex:stopIndex])
            normDistsX.append(partSum / sumX)
            startIndex = stopIndex
            stopIndex += int(lenX/nrParts)

        distributionY = self.distributionY[np.nonzero(self.distributionY)]
        distributionY = np.divide(distributionY,np.max(distributionY))

        sumY = np.sum(distributionY)
        lenY = len(distributionY)
        nrParts = 5
        startIndex = 0
        stopIndex = int(lenY / nrParts)
        nrAdditional = lenY % nrParts
        normDistsY = []
        for i in range(nrParts):
            if nrAdditional > 0:
                stopIndex += 1
                nrAdditional -= 1
            partSum = np.sum(distributionY[startIndex:stopIndex])
            normDistsY.append(partSum / sumY)
            startIndex = stopIndex
            stopIndex += int(lenY / nrParts)
        normDists = []
        normDists.extend(normDistsX)
        normDists.extend(normDistsY)

        return normDists


    """
    gebe Eckpositionen des kleinsten Rechteckes um die Zahl aus
    """
    def calcRectangle(self):
        nzX = np.nonzero(self.distributionX)
        #untere linke Ecke:
        cbl = nzX[0][0]
        #untere rechte Ecke:
        cbr = nzX[0][-1]
        nzY = np.nonzero(self.distributionY)
        #untere linke Ecke:
        cul = nzY[0][0]
        #untere rechte Ecke:
        cur = nzY[0][-1]
        return [cbl,cbr,cul,cur]




