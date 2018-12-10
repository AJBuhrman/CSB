import math as m
pods = {}
lap_count = 3
checkpoints = [
    [14000,8000],
    [5000,3000],
    [5000,8000],
    [14000,3000]]

def InitializePods():
    #angle = GetBoardAngle(checkpoints[0],checkpoints[1])
    angle = BoundAngle(GetBoardAngle(checkpoints[0],checkpoints[1]) + m.pi/2)
    pods[1] = Pod(1,[checkpoints[0][0] + -2000*m.cos(angle), checkpoints[0][1] + -2000*m.sin(angle)])
    pods[2] = Pod(2,[checkpoints[0][0] + -1000*m.cos(angle), checkpoints[0][1] + -1000*m.sin(angle)])
    pods[3] = Pod(3,[checkpoints[0][0] + 1000*m.cos(angle), checkpoints[0][1] + 1000*m.sin(angle)])
    pods[4] = Pod(4,[checkpoints[0][0] + 2000*m.cos(angle), checkpoints[0][1] + 2000*m.sin(angle)])

def GetLapCount():
    return lap_count

def GetCheckpointCount():
    return len(checkpoints)

def GetCheckpointInfo(cp_num):
    return [checkpoints[cp_num][0], checkpoints[cp_num][1]]

def GetPodInfo(num):
    x = pods[num].x
    y = pods[num].y
    vx = pods[num].vx
    vy = pods[num].vy
    hdg = pods[num].hdg
    nc = pods[num].nc
    return [x, y, vx, vy, hdg, nc]

def SendOutputs(output1, output2):
    MovePod(1,output1)
    MovePod(2,output2)
    MovePod(3)
    MovePod(4)

def MovePod(num,output = ""):
    p = pods[num]
    cp_point = [checkpoints[p.nc][0], checkpoints[p.nc][1]]

    if output != "": #One of my pods
        output = output.split()
        tx = int(output[0])
        ty = int(output[1])
        try:
            tt = int(output[2])
        except:
            tt = 0
    else: #One of their pods
        tx, ty = checkpoints[p.nc][0], checkpoints[p.nc][1]
        tt = 80

    p.hdg = BoundAngle(HowMuchTurn(p.point, [tx,ty], p.hdg) + p.hdg)

    vx = m.cos(p.hdg)*tt + p.vx #vx used to calc new x
    vy = m.sin(p.hdg)*tt + p.vy #vy used to calc new y
    p.x = p.x + vx
    p.y = p.y + vy
    p.vx = vx * 0.85
    p.vy = vy * 0.85

    if GetDistance(p.point,cp_point) < 600:
        if p.nc == GetCheckpointCount() - 1:
            p.nc = 0
        else:
            p.nc += 1

def AreWeStillPlaying():
    still_playing = True
    for i in range(1,len(pods)+1):
        if pods[i].lap > lap_count: 
            still_playing = False
            break
    return still_playing

def GetDistance(a,b):
    #Distance between two points, defined by x,y coordinates
    return m.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def HowMuchTurn(a,b,current_hdg):
    o = GetBoardAngle(a,b) - current_hdg
    n = max(ToBody(o),-18*m.pi/180)
    return min(n,18*m.pi/180)

def ToBody(angle):
    if angle > m.pi:
        return angle - 2*m.pi
    elif angle < -m.pi:
        return angle + 2*m.pi
    else:
        return angle

def BoundAngle(angle):
    if angle > 2*m.pi:
        return angle-2*m.pi
    if angle < 0:
        return angle+2*m.pi
    else:
        return angle

def GetBoardAngle(a,b):
    #Board angle between two points. Angle originates from 'a'
    #where: a = [x,y] and b = [x,y]
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    if dx == 0 and dy == 0:
        return 0
    elif dx == 0:
        if dy > 0:
            return m.pi/2
        else:
            return m.pi*3/4
    elif dy == 0:
        if dx > 0:
            return 0
        else:
            return m.pi
    elif dx < 0: #2nd and 3rd quadrants
        return m.pi + m.atan(dy/dx)
    elif dy < 0: #4th quadrant
        return 2*m.pi + m.atan(dy/dx)
    else: #1st quadrant
        return m.atan(dy/dx)

class Pod():
    def __init__(self,num,start_point):
        self.num = num
        self._x, self._y = start_point[0], start_point[1]
        self._vx, self._vy = 0, 0
        self._hdg = GetBoardAngle(self.point,checkpoints[1])
        self.prev_head = 0
        self._nc = 1
        self.lap = 0
        self.prev_cp = 0
        self.nnc = 0
        self._tx = 0 #target x
        self._ty = 0 #target y
        self._tt = 0 #target thrust
        self.message = ""
        self.next_positions = []
        self._shield = False
        self.last_shield_time = -5
        self.game_counter = 0
        self.best_gene_raw = []

    @property
    def point(self):
        return [self._x,self._y]

    @property
    def shield(self):
        return self._shield
    @shield.setter
    def shield(self,value):
        if value: self.last_shield_time = self.game_counter
        self._shield = value

    @property
    def nc(self):
        return self._nc
    @nc.setter
    def nc(self,value):
        if self._nc != value and value == 1:
            self.lap += 1
            self.prev_cp = self._nc
        elif self._nc != value:
            self.prev_cp = self._nc

        if value == len(checkpoints)-1:
            self.nnc = 0
        else:
            self.nnc = value + 1

        self._nc = value

    @property
    def x(self):
        return self._x
    @x.setter
    def x(self,value):
        self._x = int(round(value))

    @property
    def y(self):
        return self._y
    @y.setter
    def y(self,value):
        self._y = int(round(value))

    @property
    def tx(self):
        return self._tx
    @tx.setter
    def tx(self,value):
        self._tx = int(round(value))

    @property
    def ty(self):
        return self._ty
    @ty.setter
    def ty(self,value):
        self._ty = int(round(value))

    @property
    def vx(self):
        return self._vx
    @vx.setter
    def vx(self,value):
        self._vx = int(value)

    @property
    def vy(self):
        return self._vy
    @vy.setter
    def vy(self,value):
        self._vy = int(value)

    @property
    def tt(self):
        return self._tt
    @tt.setter
    def tt(self,value):
        if str(value).isalpha():
            self._tt = value
        else:
            self._tt = int(round(value))

    @property
    def hdg(self):
        return self._hdg
    @hdg.setter
    def hdg(self,value):
        self.prev_hdg = self._hdg
        self._hdg = value
