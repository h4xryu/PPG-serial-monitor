"""
Board : ART PI 
MCU : CORTEX M7, STM32H750xB
ST-Link v2

PPG 신호를 보여주는 프로그램

만든이 : 김류하

"""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
import matplotlib.pyplot as plt
import serial.tools.list_ports
import serial
import datetime as dt
import time
import datetime
import time
import pyqtgraph as pg
import pandas as pd
import numpy as np
import calendar
import random
from collections import deque
from utils import preprocess_ppg
from scipy.signal import find_peaks, butter, filtfilt


class ProcessTime(QThread):

    timeout = pyqtSignal(int)

    def __init__(self, parent):
        # QThread에서 변수 상속받음
        super().__init__(parent)
        self.start_chk = False
        self.mytimer = QTimer()
        self.mytimer.timeout.connect(self.update_time)

    def start(self):
        super().start()
        self.mytimer.start(3000)
        self.running = True

    def stop(self):
        self.running = False
        self.wait()

    def update_time(self):
        now = calendar.timegm(time.gmtime())
        self.timeout.emit(now)

    # 초 받아옴
    @staticmethod
    def get_data():
        last_x = calendar.timegm(time.gmtime())
        return last_x


class x_time(QThread):

    timeout = pyqtSignal(int)

    def __init__(self, parent):
        # QThread에서 변수 상속받음
        super().__init__(parent)
        self.start_chk = False
        self.mytimer = QTimer()
        self.mytimer.timeout.connect(self.update_time)

    def start(self):
        super().start()
        self.mytimer.start(1000)
        self.running = True

    def stop(self):
        self.running = False
        self.wait()

    def update_time(self):
        now = calendar.timegm(time.gmtime())
        self.timeout.emit(now)

    # 초 받아옴
    @staticmethod
    def get_data():
        last_x = calendar.timegm(time.gmtime())
        return last_x


class Samplingtime(QThread):

    timeout = pyqtSignal(int)

    def __init__(self, parent):
        # QThread에서 변수 상속받음
        super().__init__(parent)
        self.samptime = 10
        self.start_chk = False
        self.mytimer = QTimer()
        self.mytimer.timeout.connect(self.update_time)

    def start(self):
        super().start()
        self.mytimer.start(int(self.samptime))
        self.running = True

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def update_time(self):
        now = time.time()
        self.timeout.emit(now)

    def set_samprate(self, freq):
        self.stop()
        self.samptime = (1 / freq) * 1000
        self.start()

    # 초 받아옴
    @staticmethod
    def get_data():
        last_x = time.time()
        return last_x


class DateAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        ### 시간 시,분,초로 표현 ###
        return [time.strftime("%H:%M:%S", time.localtime(x)) for x in values]


class PPGGraphWidget(QWidget):
    def __init__(self, parent=None):
        super(PPGGraphWidget, self).__init__(parent)

        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.axitem = DateAxisItem(orientation="bottom")
        self.graphWidget = pg.PlotWidget(
            axisItems={"bottom": self.axitem}, background="#030713"
        )
        self.graphWidget.setYRange(0, 5)
        self.layout.addWidget(self.graphWidget)

    def resizeEvent(self, event):
        """윈도우 크기가 변경될 때 호출됨"""
        # 자동 리사이징을 위해 plot의 축 범위를 새로 설정
        self.graphWidget.enableAutoRange("xy", True)
        super().resizeEvent(event)


class DataMonitorWidget(QWidget):
    def __init__(self, parent=None):
        super(DataMonitorWidget, self).__init__(parent)
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.data_window = QTextEdit(self)
        self.data_window.setReadOnly(True)
        self.data_window.setFontFamily("Courier New")
        self.data_window.setStyleSheet(
            """
            QTextEdit {
                background-color: #232735;  /* 배경색 설정 */
                color: lime;             /* 폰트색상 라임색 설정 */
                font-size: 14px;         /* 폰트 크기 설정 (옵션) */
            }
        """
        )
        self.layout.addWidget(self.data_window, 1, 1)


class CaptureWidget(QWidget):
    def __init__(self, parent=None):
        super(CaptureWidget, self).__init__(parent)
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.recLabel = QLabel("Record")
        self.recLabel.setStyleSheet(
            """
            QLabel {
                color: white;             /* 폰트색상 설정 */
                font-size: 11px;         /* 폰트 크기 설정 (옵션) */
            }
        """
        )
        self.recordBtn = QPushButton(" ▶")
        self.saveBtn = QPushButton(" Save ")

        self.recordBtn.setStyleSheet(
            """
            QPushButton {
                font-size = 15px;
                border-style: outset;
                border-color: black;
                border-width: 3px;
                border-radius: 11px;      /* 둥근 모서리 */
                background-color: #EEEEEE; /* 버튼 배경색 */
                color: red;             /* 버튼 텍스트 색 */
                padding: 4px;            /* 패딩 설정 */
            }
            QPushButton:hover {
                background-color: #DDDDDD; /* 마우스 올릴 때 배경색 */
            }
            QPushButton:pressed {
                background-color: #CCCCCC; /* 클릭할 때 배경색 */
            }
        """
        )

        self.saveBtn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 4px;
                text-align: center;
                font-size: 12px;
                border-radius: 2px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        )

        self.info = QLabel("Yonsei Univ. v1.0")
        self.info.setStyleSheet(
            """
            QLabel {
                color: white;             /* 폰트색상 설정 */
                font-size: 14px;         /* 폰트 크기 설정 (옵션) */
            }
        """
        )
        self.layout.addWidget(self.recLabel, 0, 1, Qt.AlignCenter)
        self.layout.addWidget(self.recordBtn, 0, 1, Qt.AlignRight)
        self.layout.addWidget(self.saveBtn, 0, 2, Qt.AlignLeft)
        self.layout.addWidget(self.info, 1, 2, Qt.AlignBottom)


class SerialCommWidget(QWidget):
    def __init__(self, parent=None):
        super(SerialCommWidget, self).__init__(parent)
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.port_combobox = QComboBox(self)
        self.baudrate_combobox = QComboBox(self)
        self.samprate_combobox = QComboBox(self)
        self.openBtn = QPushButton(text="      Port Open      ", parent=self)
        self.openBtn.setStyleSheet(
            """
            QPushButton {
                background-color: #ADD8E6; /* 버튼 배경색 */
                color: white;             /* 버튼 텍스트 색 */
                border: 2px solid #A2C7DC; /* 테두리 설정 */
                border-radius: 1px;      /* 둥근 모서리 */
                padding: 7px;            /* 패딩 설정 */
            }
            QPushButton:hover {
                background-color: #91BEDD; /* 마우스 올릴 때 배경색 */
            }
            QPushButton:pressed {
                background-color: #76B4D4; /* 클릭할 때 배경색 */
            }
        """
        )
        self.closeBtn = QPushButton(text="      Port Close      ", parent=self)
        self.closeBtn.setStyleSheet(
            """
            QPushButton {
                background-color: #ADD8E6; /* 버튼 배경색 */
                color: white;             /* 버튼 텍스트 색 */
                border: 2px solid #A2C7DC; /* 테두리 설정 */
                border-radius: 1px;      /* 둥근 모서리 */
                padding: 7px;            /* 패딩 설정 */
            }
            QPushButton:hover {
                background-color: #91BEDD; /* 마우스 올릴 때 배경색 */
            }
            QPushButton:pressed {
                background-color: #76B4D4; /* 클릭할 때 배경색 */
            }
        """
        )
        self.startBtn = QPushButton(text="Start", parent=self)
        self.startBtn.setStyleSheet(
            """
            QPushButton {
                background-color: #ADD8E6; /* 버튼 배경색 */
                color: white;             /* 버튼 텍스트 색 */
                border: 2px solid #A2C7DC; /* 테두리 설정 */
                border-radius: 1px;      /* 둥근 모서리 */
                padding: 7px;            /* 패딩 설정 */
            }
            QPushButton:hover {
                background-color: #91BEDD; /* 마우스 올릴 때 배경색 */
            }
            QPushButton:pressed {
                background-color: #76B4D4; /* 클릭할 때 배경색 */
            }
        """
        )
        self.stopBtn = QPushButton(text="Stop", parent=self)
        self.stopBtn.setStyleSheet(
            """
            QPushButton {
                background-color: #ADD8E6; /* 버튼 배경색 */
                color: white;             /* 버튼 텍스트 색 */
                border: 2px solid #A2C7DC; /* 테두리 설정 */
                border-radius: 1px;      /* 둥근 모서리 */
                padding: 7px;            /* 패딩 설정 */
            }
            QPushButton:hover {
                background-color: #91BEDD; /* 마우스 올릴 때 배경색 */
            }
            QPushButton:pressed {
                background-color: #76B4D4; /* 클릭할 때 배경색 */
            }
        """
        )
        self.defaultBtn = QPushButton(text="Default Graph Set", parent=self)
        self.defaultBtn.setStyleSheet(
            """
            QPushButton {
                background-color: #ADD8E6; /* 버튼 배경색 */
                color: white;             /* 버튼 텍스트 색 */
                border: 2px solid #A2C7DC; /* 테두리 설정 */
                border-radius: 1px;      /* 둥근 모서리 */
                padding: 7px;            /* 패딩 설정 */
            }
            QPushButton:hover {
                background-color: #91BEDD; /* 마우스 올릴 때 배경색 */
            }
            QPushButton:pressed {
                background-color: #76B4D4; /* 클릭할 때 배경색 */
            }
        """
        )
        self.saveBtn = QPushButton(text="Save Graph Set", parent=self)
        self.saveBtn.setStyleSheet(
            """
            QPushButton {
                background-color: #ADD8E6; /* 버튼 배경색 */
                color: white;             /* 버튼 텍스트 색 */
                border: 2px solid #A2C7DC; /* 테두리 설정 */
                border-radius: 1px;      /* 둥근 모서리 */
                padding: 7px;            /* 패딩 설정 */
            }
            QPushButton:hover {
                background-color: #91BEDD; /* 마우스 올릴 때 배경색 */
            }
            QPushButton:pressed {
                background-color: #76B4D4; /* 클릭할 때 배경색 */
            }
        """
        )

        self.openBtn.setFixedSize(140, 30)
        self.closeBtn.setFixedSize(140, 30)
        self.port_combobox.setFixedSize(90, 20)
        self.baudrate_combobox.setFixedSize(90, 20)
        self.samprate_combobox.setFixedSize(90, 20)
        self.startBtn.setFixedSize(70, 60)
        self.stopBtn.setFixedSize(70, 60)

        # 시리얼 포트 리스트 불러오기
        ports = serial.tools.list_ports.comports()
        self.port_combobox.clear()
        for port in ports:
            self.port_combobox.addItem(port.device)

        graphSetBox = QVBoxLayout()
        graphSetBox.addWidget(QLabel("Graph Settings"))
        graphSetBox.addWidget(QLabel(" "))

        sTitleBox = QVBoxLayout()
        sTitleBox.addWidget(QLabel("Serial Settings"))
        sTitleBox.addWidget(QLabel(" "))

        portBox = QHBoxLayout()
        portBox.addWidget(QLabel("PortNum "))
        portBox.addWidget(self.port_combobox)

        baudBox = QHBoxLayout()
        baudBox.addWidget(QLabel("Baudrate "))
        baudBox.addWidget(self.baudrate_combobox)

        sampBox = QHBoxLayout()
        sampBox.addWidget(QLabel("Sampling Rate "))
        sampBox.addWidget(self.samprate_combobox)

        onoffBox = QHBoxLayout()
        onoffBox.addWidget(self.startBtn)
        onoffBox.addWidget(self.stopBtn)
        onoffBox.addWidget(QLabel(""))

        portbtnBox = QVBoxLayout()
        portbtnBox.addWidget(self.openBtn)
        portbtnBox.addWidget(self.closeBtn)
        portbtnBox.addWidget(QLabel(""))

        loadsaveBox = QHBoxLayout()
        loadsaveBox.addWidget(self.defaultBtn)
        loadsaveBox.addWidget(self.saveBtn)

        self.x_range = QLineEdit()
        self.x_range.setFixedSize(40, 20)
        self.y_rangemin = QLineEdit()
        self.y_rangemin.setFixedSize(40, 20)
        self.y_rangemax = QLineEdit()
        self.y_rangemax.setFixedSize(40, 20)
        axisRangeBox = QGridLayout()
        axisRangeBox.addWidget(QLabel("X Axis Range (Sec)"), 0, 0, Qt.AlignLeft)
        axisRangeBox.addWidget(self.x_range, 1, 0, Qt.AlignLeft)
        axisRangeBox.addWidget(QLabel("(1-100 Sec)"), 1, 0, Qt.AlignRight)
        axisRangeBox.addWidget(QLabel("Y Axis Range (V)"), 2, 0, Qt.AlignLeft)
        axisRangeBox.addWidget(self.y_rangemin, 3, 0, Qt.AlignLeft)
        axisRangeBox.addWidget(QLabel("~"), 3, 0, Qt.AlignCenter)
        axisRangeBox.addWidget(self.y_rangemax, 3, 0, Qt.AlignRight)

        self.ch1chk = QCheckBox(self)
        self.ch2chk = QCheckBox(self)
        self.ch3chk = QCheckBox(self)
        self.ch4chk = QCheckBox(self)

        channelBox = QGridLayout()
        channelBox.addWidget(self.ch1chk, 0, 0, Qt.AlignCenter)
        channelBox.addWidget(QLabel("Ch.1"), 0, 1, Qt.AlignCenter)
        yellowlabel = QLabel()
        yellowlabel.setStyleSheet("background-color: yellow; border:2px solid white")
        yellowlabel.setFixedSize(40, 20)
        channelBox.addWidget(yellowlabel, 0, 2, Qt.AlignCenter)

        channelBox.addWidget(self.ch2chk, 1, 0, Qt.AlignCenter)
        channelBox.addWidget(QLabel("Ch.2"), 1, 1, Qt.AlignCenter)
        redlabel = QLabel()
        redlabel.setStyleSheet("background-color: red; border:2px solid white")
        redlabel.setFixedSize(40, 20)
        channelBox.addWidget(redlabel, 1, 2, Qt.AlignCenter)

        channelBox.addWidget(self.ch3chk, 2, 0, Qt.AlignCenter)
        channelBox.addWidget(QLabel("Ch.3"), 2, 1, Qt.AlignCenter)
        bluelabel = QLabel()
        bluelabel.setStyleSheet("background-color: blue; border:2px solid white")
        bluelabel.setFixedSize(40, 20)
        channelBox.addWidget(bluelabel, 2, 2, Qt.AlignCenter)

        channelBox.addWidget(self.ch4chk, 3, 0, Qt.AlignCenter)
        channelBox.addWidget(QLabel("Ch.4"), 3, 1, Qt.AlignCenter)
        greenlabel = QLabel()
        greenlabel.setStyleSheet("background-color: green; border:2px solid white")
        greenlabel.setFixedSize(40, 20)
        channelBox.addWidget(greenlabel, 3, 2, Qt.AlignCenter)

        graphcolorBox = QGridLayout()
        graphcolorBox.addWidget(QLabel("Backgroud "), 0, 0, Qt.AlignCenter)
        graylabel = QLabel()
        graylabel.setStyleSheet("background-color: gray; border:2px solid white")
        graylabel.setFixedSize(40, 20)
        graphcolorBox.addWidget(graylabel, 0, 1, Qt.AlignRight)

        self.gridchk = QCheckBox(self)
        graphcolorBox.addWidget(self.gridchk, 1, 0, Qt.AlignLeft)
        graphcolorBox.addWidget(QLabel("Grid "), 1, 0, Qt.AlignCenter)
        graylabel = QLabel()
        graylabel.setStyleSheet("background-color: gray; border:2px solid white")
        graylabel.setFixedSize(40, 20)
        graphcolorBox.addWidget(graylabel, 1, 1, Qt.AlignRight)

        graphcolorBox.setRowStretch(0, 1)
        graphcolorBox.setRowStretch(1, 1)
        graphcolorBox.setRowStretch(2, 1)
        graphcolorBox.setRowStretch(3, 1)

        # 시리얼 포트 목록 설정
        self.load_ports()

        # 레이아웃 구성
        self.layout.addLayout(graphSetBox, 0, 4, 1, 1, Qt.AlignLeft)
        self.layout.addLayout(loadsaveBox, 0, 5, 1, 1, Qt.AlignCenter)
        self.layout.addLayout(axisRangeBox, 4, 4, 1, 1, Qt.AlignRight)
        self.layout.addLayout(channelBox, 4, 5, 1, 1, Qt.AlignRight)
        self.layout.addLayout(graphcolorBox, 4, 6, 1, 1, Qt.AlignRight)

        self.layout.addLayout(sTitleBox, 0, 0, 1, 1, Qt.AlignLeft)
        self.layout.addLayout(portBox, 2, 0, 1, 1, Qt.AlignCenter)
        self.layout.addLayout(baudBox, 3, 0, 1, 1, Qt.AlignCenter)
        self.layout.addLayout(sampBox, 2, 2, 1, 1, Qt.AlignCenter)
        self.layout.addLayout(portbtnBox, 4, 0, 1, 1, Qt.AlignCenter)
        self.layout.addLayout(onoffBox, 4, 2, 1, 1, Qt.AlignCenter)

        self.layout.setRowStretch(0, 1)
        self.layout.setRowStretch(1, 1)
        self.layout.setRowStretch(2, 1)
        self.layout.setRowStretch(3, 1)
        self.layout.setRowStretch(4, 1)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 1)
        self.layout.setColumnStretch(3, 1)
        self.layout.setColumnStretch(4, 1)
        self.layout.setColumnStretch(5, 1)

        self.layout.setContentsMargins(40, 10, 40, 20)

    def load_ports(self):
        """시리얼 포트를 로드"""
        ports = serial.tools.list_ports.comports()
        self.port_combobox.clear()
        for port in ports:
            self.port_combobox.addItem(port.device)
        self.baudrate_combobox.addItems(["9600", "57600", "115200"])
        self.samprate_combobox.addItem("100")
        self.samprate_combobox.addItem("500")
        self.samprate_combobox.addItem("1000")


class SignalThread(QThread):
    ppg_data_received = pyqtSignal(np.ndarray, np.ndarray)
    pulse_data_received = pyqtSignal(np.ndarray, np.ndarray)
    total_data_received = pyqtSignal(int, int, float, float, float)

    def __init__(self, parent, port, baud):
        super().__init__(parent)
        self.port = port
        self.baudrate = baud
        self.running = False
        self.adc0Buffer = np.array([])
        self.adc0timeBuffer = np.array([])
        self.adc1Buffer = np.array([])
        self.adc1timeBuffer = np.array([])
        self.adc0SendFlag = False
        self.adc1SendFlag = False
        self.totaltime = 0

    def start(self):
        super().start()
        self.running = True

    def stop(self):
        self.running = False

        super().quit()  # 스레드 종료 시도
        super().wait()  # 스레드가 안전하게 종료될 때까지 대기

    def sendBuffer(self):
        with serial.Serial(self.port, self.baudrate) as ser:
            if self.running:
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
                        self.adc0timeBuffer = np.append(
                            self.adc0timeBuffer, Samplingtime.get_data()
                        )

                    if len(data) >= 2 and "ADC1" in data[1]:
                        adc1Data = int(data[1].split(" ")[2])
                        # print(f"PPG Voltage : {adc1Data / 1000}[V]")

                        self.adc1Buffer = np.append(self.adc1Buffer, adc1Data)
                        self.adc1timeBuffer = np.append(
                            self.adc1timeBuffer, Samplingtime.get_data()
                        )

                        # print(self.adc1Buffer.size)

                    ### 초기화 후 전송 ###
                    if self.adc0Buffer.size >= 1:
                        # print(f"ADC0 Values : {self.adc0Buffer}")
                        self.adc0time = time.time() - start
                        # self.data_received.emit(self.adc0Buffer)
                        self.totaltime = self.totaltime + self.adc0time
                        # self.adc1Buffer = np.append(self.adc0Buffer, x_time.get_data())
                        self.pulse_data_received.emit(
                            self.adc0timeBuffer, ((self.adc0Buffer) / 1000)
                        )
                        self.adc0Buffer = np.array([])
                        self.adc0timeBuffer = np.array([])
                        self.adc0SendFlag = True

                    if self.adc1Buffer.size >= 1:
                        # print(f"ADC1 Values : {self.adc1Buffer/1000}")
                        self.adc1time = time.time() - start
                        self.ppg_data_received.emit(
                            self.adc1timeBuffer, ((self.adc1Buffer) / 1000)
                        )
                        self.totaltime = self.totaltime + self.adc1time
                        self.adc1Buffer = np.array([])
                        self.adc1timeBuffer = np.array([])
                        self.adc1SendFlag = True

                    ## 총 시간 측정 ###

                    if self.adc0SendFlag and self.adc1SendFlag:

                        self.adc0SendFlag = False
                        self.adc1SendFlag = False
                        self.total_data_received.emit(
                            adc0Data,
                            adc1Data,
                            self.adc0time,
                            self.adc1time,
                            self.totaltime,
                        )

                        # print()
                        # print("====================Times=======================")
                        # print(
                        #     f"| ADC0 Load Time : {adc0time:.5f} sec                 |"
                        # )
                        # print(
                        #     f"| ADC1 Load Time : {adc1time:.5f} sec                 |"
                        # )
                        # print(
                        #     f"| Total sending Time : {self.totaltime:.5f} sec             |"
                        # )
                        # print("================================================")
                        # print()
                        self.totaltime = 0

                except ValueError:
                    pass
                except IndexError:
                    pass
                except (serial.SerialException, ValueError) as e:
                    print(f"Error with serial connection: {e}")
                    self.running = False  # 스레드 실행 중지
                    super().quit()  # 스레드 종료 시도
                    super().wait()  # 스레드가 안전하게 종료될 때까지 대기


class MyApp(QMainWindow):

    def __init__(self):
        super(MyApp, self).__init__()

        self.setStyleSheet(
            """QMainWindow {
                background-color: #131835;
                }
                QLabel {
                    color: white;
                }"""
        )

        self.time = deque(maxlen=500)
        self.value = deque(maxlen=500)
        self.ptime = deque(maxlen=500)
        self.pvalue = deque(maxlen=500)
        self.recordBuffer = []
        self.recordtimeBuffer = []
        self.avg_interval = 0
        self.bpm = 0
        self.spo2_estimation = 0

        self.graphTimer = x_time(self)
        self.processTimer = ProcessTime(self)
        self.SamplingTimer = Samplingtime(self)
        self.recFlag = False
        self.processFlag = False
        # 메인 위젯 설정
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 그리드 레이아웃 설정
        mainGrid = QGridLayout(central_widget)

        # 각 위젯 생성
        self.ppgGraphWidget = PPGGraphWidget(self)
        self.dataMonitorWidget = DataMonitorWidget(self)
        self.serialCommWidget = SerialCommWidget(self)
        self.captureWidget = CaptureWidget(self)
        self.serialCommWidget.x_range.setText("10")
        self.serialCommWidget.y_rangemin.setText("0")
        self.serialCommWidget.y_rangemax.setText("1")
        self.serialCommWidget.ch1chk.setChecked(True)

        # 시리얼 스레드 생성
        self.port = self.serialCommWidget.port_combobox.currentText()
        self.baud = self.serialCommWidget.baudrate_combobox.currentText()
        self.serialCommWidget.startBtn.setDisabled(True)
        self.serialCommWidget.stopBtn.setDisabled(True)
        self.serialCommWidget.openBtn.clicked.connect(self.open_port)

        """상단 타이틀"""
        titleGrid = QGridLayout()
        title = QLabel("PPG Signals")
        title.setStyleSheet("font-size: 16px; font-weight: bold")
        title_data = QLabel("Data")
        titleGrid.addWidget(title, 0, 0, 1, 1)
        mainGrid.addLayout(titleGrid, 0, 0, 1, 1, Qt.AlignCenter)
        mainGrid.addWidget(title_data, 0, 1, 1, 1, Qt.AlignCenter)

        # 각 위젯 배치
        mainGrid.addWidget(self.ppgGraphWidget, 1, 0)
        mainGrid.addWidget(self.dataMonitorWidget, 1, 1)
        mainGrid.addWidget(self.serialCommWidget, 2, 0, 1, 1)
        mainGrid.addWidget(self.captureWidget, 2, 1)
        mainGrid.setColumnStretch(0, 4)
        mainGrid.setColumnStretch(1, 1)

        # 버튼 클릭 시그널 연결
        self.serialCommWidget.startBtn.clicked.connect(self.start_serial)
        self.serialCommWidget.stopBtn.clicked.connect(self.stop_serial)
        self.captureWidget.recordBtn.clicked.connect(self.set_recFlag)
        self.captureWidget.saveBtn.clicked.connect(self.savePPGSignal)

        # 창 설정
        self.setWindowTitle("PPG Signal Monitor")
        self.setGeometry(300, 300, 1280, 720)

    def open_port(self):
        self.port = self.serialCommWidget.port_combobox.currentText()
        self.baud = self.serialCommWidget.baudrate_combobox.currentText()
        self.serial_thread = SignalThread(self, self.port, self.baud)
        if self.baud is None:
            raise ValueError("Baudrate is not set. Please set a valid baudrate.")

        self.serialCommWidget.startBtn.setEnabled(True)
        self.serialCommWidget.stopBtn.setEnabled(True)

        # 시리얼 스레드의 시그널 슬롯 연결
        self.graphTimer.timeout.connect(self.update_ppgGraph)
        self.SamplingTimer.timeout.connect(self.serial_thread.sendBuffer)
        self.serial_thread.ppg_data_received.connect(self.get_ppg_values)
        self.serial_thread.pulse_data_received.connect(self.get_pulse_values)
        self.serial_thread.total_data_received.connect(self.update_monitor)
        self.processTimer.timeout.connect(self.process)

        self.serialCommWidget.samprate_combobox.currentIndexChanged.connect(
            self.setSamprate
        )

    def process(self):
        self.processFlag = True

        peak_time = [
            (
                float((int(self.ptime[i] * 100) % 6000) / 100)
                if self.pvalue[i] > 15
                else 0
            )
            for i in range(len(self.ptime))
        ]

        blocks = []
        current_block = []

        """BPM 계산"""

        for value in peak_time:
            if value != 0:
                current_block.append(value)
            elif current_block:
                blocks.append(current_block)
                current_block = []
                if current_block:
                    blocks.append(current_block)  # 마지막 블록 추가

        first_values = [block[0] for block in blocks]
        intervals = np.diff(first_values)  # 첫 번째 값들 간의 차이 계산
        self.avg_interval = np.mean(intervals)

        """산소포화도 추정"""

        self.processFlag = False

    def set_recFlag(self):
        self.recFlag = True

    def savePPGSignal(self):

        samprate = int(self.serialCommWidget.samprate_combobox.currentText())
        preprocessed = preprocess_ppg(self.recordBuffer, samprate)

        if len(self.recordtimeBuffer) != len(preprocessed):
            # np.linspace를 사용하여 preprocessed의 길이에 맞춘 새로운 타임 버퍼 생성
            new_time_buffer = np.linspace(
                self.recordtimeBuffer[0], self.recordtimeBuffer[-1], len(preprocessed)
            )
        else:
            new_time_buffer = self.recordtimeBuffer

        self.saveFlag = False
        plt.title("PPG Signals")
        plt.ylim(-5, 5)
        plt.plot(new_time_buffer, preprocessed)
        plt.savefig("F:\PPG_results\ppg_waves.png")
        plt.show()
        timestamp = np.array(self.recordtimeBuffer)
        df = pd.DataFrame(
            {
                "Time": list(25569 + (timestamp + 32400) / 86400),
                "Voltage": self.recordBuffer,
            }
        )
        df.to_excel("F:\PPG_results\data.xlsx", index=False)
        self.recordtimeBuffer = []
        self.recordBuffer = []
        self.recFlag = False

    def get_pulse_values(self, time, value):
        self.ptime.extend(time)
        self.pvalue.extend(value)

    def get_ppg_values(self, time, value):
        self.time.extend(time)
        self.value.extend(value)
        if self.recFlag:
            self.recordtimeBuffer.append(time)
            self.recordBuffer.append(value)

    def setSamprate(self):
        freq = self.serialCommWidget.samprate_combobox.currentText()
        self.serial_thread.stop()
        self.SamplingTimer.set_samprate(float(freq))
        self.serial_thread.start()

    def update_ppgGraph(self):
        """그래프를 업데이트하는 메서드"""
        scailed_pvalue = [
            (
                float(self.serialCommWidget.y_rangemax.text())
                if value >= 15
                else float(self.serialCommWidget.y_rangemin.text())
            )
            for value in self.pvalue
        ]
        if self.serialCommWidget.ch1chk.isChecked():
            self.ppgGraphWidget.graphWidget.plot(
                x=self.time, y=self.value, pen=pg.mkPen("yellow", width=2)
            )

        self.serialCommWidget.y_rangemin.text()

        if self.serialCommWidget.ch2chk.isChecked():
            self.ppgGraphWidget.graphWidget.plot(
                x=self.ptime, y=scailed_pvalue, pen=pg.mkPen("red", width=2)
            )

        self.graph_settings()

    def update_monitor(self, adc0, adc1, adc0time, adc1time, totaltime):
        """시리얼 데이터를 텍스트로 표시하는 메서드"""

        adjusted = (
            float(self.serialCommWidget.y_rangemax.text())
            if adc0 / 1000 >= 15
            else float(self.serialCommWidget.y_rangemin.text())
        )

        self.ppg_info = (
            "Board : ART PI\n"
            "MCU : STM32H750xB (CORTEX M7)\n"
            "ST-Link v2\n\n"
            f"PPG : {adc1 / 1000:.3f} V\n"
            f"PULSE : {adc0 / 1000:.3f} V\n"
            f"Adjusted pulse : {adjusted} V\n\n"
            f"ADC0 Load time : {adc1time:.3f} sec\n"
            f"ADC1 Load time : {adc0time:.3f} sec\n"
            f"Total time : {totaltime:.3f} sec\n"
        )

        if self.recFlag:
            self.ppg_info += "\n\n\nRecording...\n"

            try:
                # 평균 주기가 0 이상인 경우에만 BPM 계산
                if self.avg_interval > 0:
                    self.bpm = 60 / self.avg_interval
                    # print("bpm", bpm)
                    # BPM이 200 미만일 경우에만 유효한 값으로 처리
                    if self.bpm < 200:
                        self.ppg_info += f"BPM : {int(self.bpm)}\n"
                    else:
                        self.ppg_info += "BPM: N/A (Invalid BPM)\n"
                else:
                    self.ppg_info += "BPM: N/A (Invalid interval)\n"

                self.ppg_info += f"SPO2 Estimation: {self.spo2_estimation}%\n"
            except:
                pass

        self.dataMonitorWidget.data_window.setText(self.ppg_info)

    def graph_settings(self):

        range = int(self.serialCommWidget.x_range.text())
        if range <= 100 and range > 0:
            self.ppgGraphWidget.graphWidget.setXRange(time.time() - range, time.time())
        else:
            self.serialCommWidget.x_range.setText("10")
            self.ppgGraphWidget.graphWidget.setXRange(time.time() - 10, time.time())

        yminRange = float(self.serialCommWidget.y_rangemin.text())
        ymaxRange = float(self.serialCommWidget.y_rangemax.text())

        if yminRange >= 0 and ymaxRange > yminRange:
            self.ppgGraphWidget.graphWidget.setYRange(yminRange, ymaxRange)
        else:
            self.ppgGraphWidget.graphWidget.setYRange(0, 1)

        # 레코딩 설정
        if self.recFlag:
            pass

    def start_serial(self):
        self.graphTimer.timeout.connect(self.update_ppgGraph)
        self.serial_thread.ppg_data_received.connect(self.get_ppg_values)
        self.serial_thread.pulse_data_received.connect(self.get_pulse_values)
        self.serial_thread.start()
        self.SamplingTimer.start()
        self.graphTimer.start()
        self.processTimer.start()
        self.serialCommWidget.port_combobox.setDisabled(True)
        self.serialCommWidget.baudrate_combobox.setDisabled(True)

    def stop_serial(self):
        self.serial_thread.stop()
        self.graphTimer.timeout.disconnect()
        self.serial_thread.ppg_data_received.disconnect()
        self.serial_thread.pulse_data_received.disconnect()
        self.serial_thread.running = False
        self.serialCommWidget.port_combobox.setEnabled(True)
        self.serialCommWidget.baudrate_combobox.setEnabled(True)

        self.time = deque(maxlen=500)
        self.value = deque(maxlen=500)
        self.ptime = deque(maxlen=500)
        self.pvalue = deque(maxlen=500)

    def closeEvent(self, event):
        """창이 닫힐 때 시리얼 스레드를 중지"""
        self.serial_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    app.exec_()
