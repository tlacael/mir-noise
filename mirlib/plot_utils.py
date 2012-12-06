from scipy import *
from matplotlib.pyplot import *

plotColors = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'w-']
def GetPlotColor(ind):
    return plotColors[ind]

class DataManager:
    pass
    ''' This class holds data sets for plotting so that you can refresh plots easily.

    You can use it to update datasets, and then refresh / reconstruct the plots in the
    PlotManager class. 

    self.figures is a list of each individual figure in the plot.

    Each figure in self.figures is list of axes. Each axis is a list of datasets. '''
    
    def __init__(self):
        self.figures = []

    def Setup(self, nFigures, nSubFiguresEach):
        for fig in range(nFigures):
            i = self.NewFigure()

            for subfig in range(nSubFiguresEach):
                self.AddAxis(i)

    def IsValidFigure(self, figureID):
        if figureID < len(self.figures):
            return True
        else:
            return False

    def IsValidAxis(self, figureID, axisID):
        if self.IsValidFigure(figureID) and axisID < len(self.figures[figureID]):
            return True
        else:
            return False

    def IsValidData(self, figureID, axisID, i):
        if self.IsValidAxis(figureID, axisID) and i < len(self.figures[figureID][axisID]):
            return True
        else:
            return False

    def AddData(self, figureID, axisID, data):
        if self.IsValidAxis(figureID, axisID):
            self.figures[figureID][axisID].append(data)
        else:
            raise ValueError("Figure or Axis Index out of range")

    def ModifyData(self, figureID, axisID, dataID, data):
        if self.IsValidData(figureID, axisID, dataID):
            self.figures[figureID][axisID][dataID] = data
        elif self.IsValidAxis(figureID, axisID) and self.GetDataCount(figureID, axisID) is dataID:
            self.AddData(figureID, axisID, data)
        else:
            raise ValueError("Index out of range: [%d %d %d]" % (figureID, axisID, dataID))

    def GetData(self, figureID, axisID, dataID):
        if self.IsValidData(figureID, axisID, dataID):
            return self.figures[figureID][axisID][dataID]
        else:
            raise ValueError("Index out of range...")

    def GetDataCount(self, figureID, axisID):
        if self.IsValidAxis(figureID, axisID):
            return len(self.figures[figureID][axisID])
        else:
            raise ValueError("Index out of range...")

    def GetAxisCount(self, figureID):
        if self.IsValidFigure(figureID):
            return len(self.figures[figureID])
        else:
            raise ValueError("Figure index out of range...")

    def GetFigureCount(self):
        return len(self.figures)

    def NewFigure(self):
        self.figures.append([])
        return len(self.figures) - 1

    def AddAxis(self, figureID):
        self.figures[figureID].append([])
        return len(self.figures[figureID]) - 1

class PlotManager:
    def __init__(self, dataMgr):
        self.data = dataMgr
        self.currentFig = 0
        self.subplots = []

    def RefreshFigures(self):
        close('all')

        for figID in range(self.data.GetFigureCount()):
            self.NewFigure(figID + 1)

            subplots = self.data.GetAxisCount(figID)
            for subID in range(subplots):
                ax = self.NewSubplot(subplots, subID)

                dataCount = self.data.GetDataCount(figID, subID)
                for dataID in range(dataCount):
                    self.DrawData( ax, self.data.GetData(figID, subID, dataID), color=GetPlotColor(dataID) )
    

    def show(self):
        print "Drawing..."
        self.Draw()

    def DrawData(self, axis, data, color='b-', title='', xlabel='', ylabel='', label=''):
        if len(data.shape) > 1 and data.shape[1] > 1:
            print data.shape, max(data[0]), min(data[1]), max(data[1])
            axis.plot(data[0], data[1], color)
            axis.set_xlim( 0, max(data[0]) )
            axis.set_ylim( min(data[1]), max(data[1]) )
        else:
            print data.shape, min(data), max(data)
            axis.plot(data, color=color)
            axis.set_title(title)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.set_xlim( 0, max(data))
            axis.set_ylim( min(data), max(data))

    def NewFigure(self, id):
        print "Creating Figure:", id
        self.currentFig = figure()

    def NewSubplot(self, subplotcount, subplotID):
        print "Creating Subplot:", subplotcount, subplotID
        ax = self.currentFig.add_subplot(subplotcount, 1, subplotID + 1)
        self.subplots.append(ax)
        return ax
        
    def PlotFreq(self, data, title='', xlabel='', ylabel=''):
        self.currentAx.imshow(data.transpose(), interpolation='nearest', aspect='auto')
        self.currentAx.set_title(title)
        self.currentAx.set_xlabel(xlabel)
        self.currentAx.set_ylabel(ylabel)

    def Draw(self):
        draw()
        #self.currentFig.canvas.draw()
        #show()

    def Show(self):
        show()

