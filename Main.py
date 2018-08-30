import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

from Cashbox import cashbox

class App(QWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'Cash Box Detector'
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 200
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        button = QPushButton('Start', self)
        button.setToolTip('This is an example button')
        button.move(100,70) 
        button.clicked.connect(self.on_click)

        self.show()
 
    @pyqtSlot()
    def on_click(self):
        ## variables
        cashbox()
        
if __name__ == '__main__':
    ## QtApp
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
