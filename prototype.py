import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QMainWindow,
    QStatusBar,
    QComboBox,
    QLineEdit,
    QTextEdit,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
import serial.tools.list_ports

import datetime
import time
import pyqtgraph as pg
import pandas as pd
import numpy as np

import calendar
import random


class x_time(QThread):

    timeout = pyqtSignal(pd.DataFrame)

    def __init__(self, parent):

        # QThread에서 변수 상속받음

        super().__init__(parent)

        # 필요한 변수 사용, self는 자기자신 또는 캡슐화역할 외부에서 수정불가로하게 해줌

        self.start_chk = False

        self.sec_per_day = 3600 * 24

        now = dt.datetime.now()

        self.x = [calendar.timegm(time.gmtime())]

    # 초 받아옴

    @staticmethod  # 정적 메소드
    def get_data():

        # 우선 아두이노 연결이 안되어있으니 임시로 랜덤값으로 배정함

        # 1초마다 리스트에 값 추가됨 이걸 그래프로 옮겨서 그리는 작업

        mytimer = QTimer()

        mytimer.start(1000)  # 1초마다 차트 갱신 위함..

        cnt_x = 0  # 초 세기

        last_x = calendar.timegm(time.gmtime()) + cnt_x  # 1초뒤의 시간을 계속 더함

        cnt_x = cnt_x + 1  # 초 증가

        return last_x


class DateAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        ### 시간 시,분,초로 표현 ###
        return [time.strftime("%H:%M:%S", time.localtime(x)) for x in values]


class SignalThread(QThread):
    data_received = pyqtSignal(np.ndarray)

    def __init__(self, parent, port, baud):
        super().__init__(parent)

        self.port = port
        self.baudrate = baud
        self.running = False
        self.adc0Buffer = np.array([])
        self.adc1Buffer = np.array([])
        self.adc0SendFlag = False
        self.adc1SendFlag = False
        self.totaltime = 0

    def start(self):
        super().start()
        self.running = True

    def stop(self):
        self.running = False
        self.wait()

    def sendBuffer(self):
        with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
            while self.running:
                try:

                    start = time.time()
                    data = ser.readline().decode().rstrip("\r\n")
                    if len(data) > 0:
                        data = data.split(",")

                    # print(data)
                    # print(len(data))

                    ### 버퍼에 값 저장 ###
                    if len(data) >= 2 and "ADC0" in data[0]:
                        adc0Data = int(data[0].split(" ")[1])
                        # print(f"PULSE Volatage : {adc0Data / 1000}[V]")
                        self.adc0Buffer = np.append(self.adc0Buffer, adc0Data)

                    if len(data) >= 2 and "ADC1" in data[1]:
                        adc1Data = int(data[1].split(" ")[2])
                        # print(f"PPG Voltage : {adc1Data / 1000}[V]")
                        self.adc1Buffer = np.append(self.adc1Buffer, adc1Data)

                    ### 초기화 후 전송 ###
                    if self.adc0Buffer.size == 50:
                        print(f"ADC0 Values : {self.adc0Buffer}")
                        adc0time = time.time() - start
                        self.data_received.emit(self.adc0Buffer)
                        self.totaltime = self.totaltime + adc0time
                        self.adc0Buffer = np.array([])
                        self.adc0SendFlag = True

                    if self.adc1Buffer.size == 50:
                        print(f"ADC1 Values : {self.adc1Buffer}")
                        adc1time = time.time() - start
                        self.data_received.emit(self.adc1Buffer)
                        self.totaltime = self.totaltime + adc1time
                        self.adc1Buffer = np.array([])
                        self.adc1SendFlag = True

                    ## 총 시간 측정 ###

                    if self.adc0SendFlag and self.adc1SendFlag:

                        self.adc0SendFlag = False
                        self.adc1SendFlag = False

                        print()
                        print("====================Times=======================")
                        print(
                            f"| ADC0 Load Time : {adc0time:.5f} sec                 |"
                        )
                        print(
                            f"| ADC1 Load Time : {adc1time:.5f} sec                 |"
                        )
                        print(
                            f"| Total sending Time : {self.totaltime:.5f} sec             |"
                        )
                        print("================================================")
                        print()
                        self.totaltime = 0

                except ValueError:
                    pass
                except IndexError:
                    pass
                except serial.SerialException as e:
                    print(f"Error opening serial port: {e}")
                finally:
                    if self.ser:
                        self.ser.close()

    def run(self):
        self.sendBuffer()

    def chk_run(self) -> bool:
        return self.running

    def stop(self):
        self.running = False
        self.wait()


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.text_edit = []
        self.serialcommGrid = QGridLayout()
        self.initUI()

        ### 디버깅 ###
        # self.testThread = SignalThread(self, "COM3", 57600)
        # self.testThread.start()

        self.serial_thread = SignalThread(self, "COM3", 57600)

    def initUI(self):
        # 그리드 레이아웃 설정
        mainGrid = QGridLayout()
        self.setLayout(mainGrid)

        # 레이아웃 이니셜라이징
        self.createTitle(mainGrid)
        self.initPPGGraph(mainGrid)
        self.initPulseGraph(mainGrid)
        self.initDataTable(mainGrid)
        self.initSerialComm(mainGrid)
        self.initSettings(mainGrid)

        # 창 설정
        self.setWindowTitle("PPG Signal")
        self.setGeometry(300, 300, 1280, 720)
        self.show()

    # =============================레이아웃 이니셜라이징==============================
    def createTitle(self, grid):
        """상단 타이틀"""
        titleGrid = QGridLayout()
        title = QLabel("PPG Signals")
        titleGrid.addWidget(title, 0, 0, 1, 1)
        grid.addLayout(titleGrid, 0, 0, 1, 2, Qt.AlignCenter)

    def initPPGGraph(self, grid):
        """PPG 그래프"""
        PPGGraphGrid = QGridLayout()
        axitem = DateAxisItem(orientation="bottom")
        self.graphWidget = pg.PlotWidget(
            axisItems={"bottom": axitem}, background="gray"
        )

        PPGGraphGrid.addWidget(self.graphWidget, 1, 0)
        grid.addLayout(PPGGraphGrid, 1, 0)

    def initPulseGraph(self, grid):
        """펄스 그래프"""
        pass

    def initDataTable(self, grid):
        """데이터 테이블"""
        dataTableGrid = QGridLayout()
        self.data_window = QTextEdit()
        self.data_window.setReadOnly(True)
        self.data_window.setFontFamily("Courier New")

        dataTableGrid.addWidget(self.data_window, 1, 1)
        grid.addLayout(dataTableGrid, 1, 1)

    def initSerialComm(self, grid):
        """통신 설정"""
        self.port_combobox = QComboBox(self)
        self.baudrate_combobox = QComboBox(self)
        self.samprate_combobox = QComboBox(self)
        self.openBtn = QPushButton(text="Port Open", parent=self)
        self.closeBtn = QPushButton(text="Port Close", parent=self)
        self.startBtn = QPushButton(text="Start", parent=self)
        self.stopBtn = QPushButton(text="Stop", parent=self)

        # self.openBtn.setFixedSize(140, 30)
        # self.closeBtn.setFixedSize(140, 30)
        # self.port_combobox.setFixedSize(90, 20)
        # self.baudrate_combobox.setFixedSize(90, 20)
        # self.samprate_combobox.setFixedSize(90, 20)
        # self.startBtn.setFixedSize(70, 60)
        # self.stopBtn.setFixedSize(70, 60)

        # 시리얼 포트 리스트 불러오기
        ports = serial.tools.list_ports.comports()
        self.port_combobox.clear()
        for port in ports:
            self.port_combobox.addItem(port.device)
        self.baudrate_combobox.addItem("9600")
        self.baudrate_combobox.addItem("57600")
        self.baudrate_combobox.addItem("115200")
        self.samprate_combobox.addItem("500")

        sTitleBox = QVBoxLayout()
        sTitleBox.addWidget(QLabel("Serial Settings"))
        sTitleBox.addWidget(QLabel(" "))
        portBox = QHBoxLayout(self)
        portBox.addWidget(QLabel("PortNum "))
        portBox.addWidget(self.port_combobox)
        baudBox = QHBoxLayout(self)
        baudBox.addWidget(QLabel("Baudrate "))
        baudBox.addWidget(self.baudrate_combobox)
        sampBox = QHBoxLayout(self)
        sampBox.addWidget(QLabel("Sampling Rate "))
        sampBox.addWidget(self.samprate_combobox)
        onoffBox = QHBoxLayout(self)
        onoffBox.addWidget(self.startBtn)
        onoffBox.addWidget(self.stopBtn)
        onoffBox.addWidget(QLabel(""))
        portbtnBox = QVBoxLayout(self)
        portbtnBox.addWidget(self.openBtn)
        portbtnBox.addWidget(self.closeBtn)
        portbtnBox.addWidget(QLabel(""))

        self.serialcommGrid.addLayout(sTitleBox, 0, 0, 3, 1, Qt.AlignCenter)
        self.serialcommGrid.addLayout(portBox, 2, 0, 1, 2, Qt.AlignCenter)
        self.serialcommGrid.addLayout(baudBox, 3, 0, 1, 2, Qt.AlignCenter)
        self.serialcommGrid.addLayout(sampBox, 2, 2, 1, 2, Qt.AlignCenter)
        self.serialcommGrid.addLayout(portbtnBox, 4, 0, 2, 2, Qt.AlignCenter)
        self.serialcommGrid.addLayout(onoffBox, 4, 2, 1, 2, Qt.AlignCenter)

    def initSettings(self, grid):
        """그래프 세팅"""
        self.text_input = QLineEdit(self)
        self.defaultBtn = QPushButton(self)
        self.saveBtn = QPushButton(self)

        graphSetBox = QVBoxLayout()
        graphSetBox.addWidget(QLabel("Graph Settings"))
        graphSetBox.addWidget(QLabel(" "))
        loadsaveBox = QHBoxLayout()
        loadsaveBox.addWidget(self.defaultBtn)
        loadsaveBox.addWidget(self.saveBtn)

        self.serialcommGrid.addLayout(graphSetBox, 0, 4, 3, 1, Qt.AlignCenter)
        self.serialcommGrid.addLayout(loadsaveBox, 0, 5, 3, 2, Qt.AlignCenter)
        grid.addLayout(self.serialcommGrid, 2, 0, 1, 1)

    # =============================시리얼 통신=====================================
    def start_serial(self):
        """시리얼 통신 세팅"""
        selected_port = self.port_combobox.currentText()

        if selected_port:
            self.serial_thread.data_received.connect(self.update_text_edit)
            self.serial_thread.start()

            # 버튼 상태 변경
            self.startBtn.setDisabled(True)
            self.stopBtn.setDisabled(False)

    def stop_serial(self):
        """시리얼 통신을 종료"""
        # if self.serial_thread:
        #     self.serial_thread.stop()
        #     self.serial_thread = None

        #     # 버튼 상태 변경
        #     self.start_button.setDisabled(False)
        #     self.stop_button.setDisabled(True)

    def update_text_edit(self, data):
        """시리얼 데이터를 텍스트 창에 출력"""
        self.text_edit.append(data)

    def closeEvent(self, event):
        """창이 닫힐 때 스레드를 종료"""
        if self.serial_thread:
            self.serial_thread.stop()
        event.accept()

    def resizeEvent(self, event):
        """윈도우 크기가 변경될 때 호출됨"""
        # 자동 리사이징을 위해 plot의 축 범위를 새로 설정
        self.graphWidget.enableAutoRange("xy", True)
        super().resizeEvent(event)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
