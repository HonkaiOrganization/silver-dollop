import gui
import sys

def main():
    app = gui.QApplication(sys.argv)
    
    window = gui.MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()