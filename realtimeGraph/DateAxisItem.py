import time
import pyqtgraph as pg


class DateAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        ### 시간 시,분,초로 표현 ###
        return [time.strftime("%H:%M:%S", time.localtime(x)) for x in values]
