import matplotlib.pyplot as plt
import numpy as np
import math
import time

class Bezier:
    def __init__(self, points, interpolationNum):
        self.dimension = points.shape[1]
        self.order = points.shape[0]-1
        self.interNum = interpolationNum
        self.pointsNum = points.shape[0]
        self.Points = points

    def getBezierPoints(self, method):
        if method==0:
            return self.DigitalAlgo()
        if method==1:
            return self.DeCasteljauAlgo()
        if method==2:
            return self.DeCasteljauAlgo2()

    def DigitalAlgo(self):
        PB = np.zeros((self.pointsNum, self.dimension))
        pis = []
        n = self.order
        for t in np.arange(0, 1+1/self.interNum, 1/self.interNum):
            for i in range(0, self.pointsNum):
                PB[i] = (math.factorial(n) / (math.factorial(i) * math.factorial(n-i))) * ((1-t)**(n-i)) * (t**i) * self.Points[i]
            pi = np.sum(PB, axis=0)
            pis.append(pi)
        return np.array(pis)

    # 递归转顺序实现（优化内存）
    def DeCasteljauAlgo(self):
        pis = []
        for t in np.arange(0, 1+1/self.interNum, 1/self.interNum):
            Att = self.Points
            for i in np.arange(0, self.order):
                for j in np.arange(0, self.order-i):
                    Att[j] = (1-t)*Att[j] + t*Att[j+1]
            pis.append(Att[0].tolist())
        return np.array(pis)

    # 递归方法实现
    def DeCasteljauAlgo2(self):
        pis = []
        for t in np.arange(0, 1+1/self.interNum, 1/self.interNum):
            pis.append(self.DAlgoNd(t, self.order, 0))
        return np.array(pis)

    def DAlgoNd(self, t, n, p):
        if n>1:
            plist = []
            B0 = self.DAlgoNd(t, n-1, p)
            B1 = self.DAlgoNd(t, n-1, p+1)
            B = (1-t)*B0 + t*B1
            return B
        else:
            return (1-t)*self.Points[p] + t*self.Points[p+1]


class Line:
    def __init__(self,Points,InterpolationNum):
        self.demension=Points.shape[1]    # 点的维数
        self.segmentNum=InterpolationNum-1 # 段数
        self.num=InterpolationNum         # 单段插补(点)数
        self.pointsNum=Points.shape[0]   # 点的个数
        self.Points=Points                # 所有点信息
        
    def getLinePoints(self):
        # 每一段的插补点
        pis=np.array(self.Points[0])
        # i是当前段
        for i in range(0,self.pointsNum-1):
            sp=self.Points[i]
            ep=self.Points[i+1]
            dp=(ep-sp)/(self.segmentNum)# 当前段每个维度最小位移
            for i in range(1,self.num):
                pi=sp+i*dp
                pis=np.vstack((pis,pi))         
        return pis

points=np.array([
    [1,3,0],
    [1.5,1,0],
    [4,2,0],
    [4,3,4],
    # [2,3,11],
    # [5,5,9]
    ])
# points=np.array([
#     [0.0,0.0],
#     [1.0,0.0],
#     [1.0,1.0],
#     [0.0,1.0],
#     ])
    


if __name__ == '__main__':
    
    if points.shape[1]==3:
        fig=plt.figure()
        ax = fig.gca(projection='3d')
        
        # 标记控制点
        for i in range(0,points.shape[0]):
            ax.scatter(points[i][0],points[i][1],points[i][2],marker='o',color='r')
            ax.text(points[i][0],points[i][1],points[i][2],i,size=12)
        
        # 直线连接控制点
        l=Line(points,100)
        pl=l.getLinePoints()
        ax.plot3D(pl[:,0],pl[:,1],pl[:,2],color='k')
        
        # 贝塞尔曲线连接控制点
        bz=Bezier(points,1000)

        # start = time.time()
        # matpi=bz.getBezierPoints(0)
        # end = time.time()
        # print(end - start)

        # start = time.time()
        # matpi=bz.getBezierPoints(1)
        # end = time.time() 
        # print(end - start)

        start = time.time()
        matpi=bz.getBezierPoints(2)
        end = time.time() 
        print(end - start)
        
        ax.plot3D(matpi[:,0],matpi[:,1],matpi[:,2],color='r')
        plt.show()

    if points.shape[1]==2:  
        # 标记控制点
        for i in range(0,points.shape[0]):
                plt.scatter(points[i][0],points[i][1],marker='o',color='r')
                plt.text(points[i][0],points[i][1],i,size=12)
                
        # 直线连接控制点
        l=Line(points,1000)
        pl=l.getLinePoints()
        plt.plot(pl[:,0],pl[:,1],color='k')
        
        # 贝塞尔曲线连接控制点
        bz=Bezier(points,1000)
        matpi=bz.getBezierPoints(1)
        plt.plot(matpi[:,0],matpi[:,1],color='r')
        plt.show()