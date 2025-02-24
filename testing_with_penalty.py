"""
---------Mathematical Model Formutlation--------
--------Exact Method Solution with Gurobi-------

With Order Schedule FCFS Penalty 
Made in completion for thesis research (2022)
"""

import random
import math
from gurobipy import Model, GRB, quicksum 

#defininng model
mdl = Model("sbsrsproblem2 (notation prob 7)")

#size variables
n = 10 # number of slots/locations
m = 3 # number of tiers
v = 3 # number of SKU
qq = 9 # the total of all demand tasks
cl = 3  # cluster of orders
tp = 1 # time penalty(s)

#system parameter
dl = 1 # distance length each coloumn (m)
vx = 2 # speed of shuttle (m/s)
ax = 1 # acceleration (m^2/s)
dh = 0.6 # distance height each tier (m)
vy = 1 # speed of the lift (m/s)
ay = 1 # acceleration of the lift (m^2/s)

#size sets variables
N = [i for i in range(1, n+1)] #a set of location
M = [i for i in range(1, m+1)] #a set of tier
V = [0] + N #a set of location with I/O point
U = [i for i in range(1, v+1)] #a set of sku
Q = [i for i in range (1, qq+1)] #a set of demand
B = [i for i in range (0, qq+1)] #a set of iteration with 0 iteration
A = [i for i in range (0,cl)] # a set of cluster
C = [i for i in range (1, qq+1-cl+1)] # 1 until 8
D = [i for i in range (qq+1-cl+1,qq+1)] #a set of additional task in cluster
#c = [(i,j) for i in range(0, n+1) for j in range (0, m+1)] #cij as a tuple??
c = {}
p = {}

#cost variable (his should be time later on)    
for j in M :
    if (j-1)*dh > (vy**2)/ay :
        for i in N :
            if i*dl > (vx**2)/ax :
                c[i,j] = 2*((2*vx/ax) + ((i*dl) - ((vx**2)/ax))/vx + (2*vy/ay) + (((j-1)*dh) - ((vy**2)/ay))/vy) #cost if reach max velocity
            if i*dl <= (vx**2)/ax :
                c[i,j] = 2*(2*math.sqrt((i*dl)/ax) + (2*vy/ay) + (((j-1)*dh) - ((vy**2)/ay))/vy) #cost if havent reach max velocity
    if (j-1)*dh <= (vy**2)/ay :
        for i in N :
            if i*dl > (vx**2)/ax :
                c[i,j] = 2*((2*vx/ax) + ((i*dl) - ((vx**2)/ax))/vx + (2*math.sqrt(((j-1)*dh)/ay))) #cost if reach max velocity
            if i*dl <= (vx**2)/ax :
                c[i,j] = 2*(2*math.sqrt((i*dl)/ax) + (2*math.sqrt(((j-1)*dh)/ay))) #cost if havent reach max velocity

#c = {(i): 2*(V[i]-0) for i in N} #cost

#demand of each task (storing or retrieving or do nothing)
q = {
     (1, 1): 0,(1, 2): 0,(1, 3): 1,
     (2, 1): 0,(2, 2): 0,(2, 3): 1,
     (3, 1): 0,(3, 2): 1,(3, 3): 0,
     (4, 1): 0,(4, 2): 0,(4, 3): 1,
     (5, 1): 0,(5, 2): 0,(5, 3): 1,
     (6, 1): 0,(6, 2): 0,(6, 3): 1,
     (7, 1): 0,(7, 2): 0,(7, 3): -1,
     (8, 1): 0,(8, 2): -1,(8, 3): 0,
     (9, 1): 0,(9, 2): 0,(9, 3): -1,
    }
S = [1,4,7]

#penalty cost 
for g in Q :
    for f in Q :
        if f > g :
            p[g,f] = abs(f-g)*tp
        if f <= g :
            p[g,f] = 0

#set variables for helping decision variable
H = [(i, j, u, f) for i in N for j in M for f in Q for u in U]
F = [(g, u, f) for g in Q for u in U for f in Q]
G = [(i, j, u, f) for u in U for i in N for j in M for f in B]
P = [(i, j, g, u, f) for i in N for j in M for f in Q for g in Q for u in U]
T = [(f) for f in Q]

#desicion variables
x = mdl.addVars(H, vtype=GRB.BINARY, name="x")  #desicion variable storing (xif)
y = mdl.addVars(F, vtype=GRB.BINARY, name="y")  #desicion variable retrieving (yif)
sc= mdl.addVars(G, vtype=GRB.BINARY, name="sc") #storage condition update variable (SCif) 
z = mdl.addVars(P, vtype=GRB.BINARY, name="z") #for Z additional variable for linearized version

#additional variable for linearization
b = mdl.addVars(T, vtype=GRB.BINARY, name="b") #additional variable for continuity

#setting objfunct 
mdl.modelSense = GRB.MINIMIZE
mdl.setObjective(quicksum(x[i, j, u, f]*c[i, j] for i in N for j in M for f in Q for u in U) + quicksum(y[g,u,f]*p[g,f] for g in Q for u in U for f in Q))
#mdl.setObjective(quicksum(x[i, j, u, f]*c[i, j] for i in N for j in M for f in Q for u in U))

#sttingup constraints
mdl.addConstrs(quicksum(x[i, j, u, f] for u in U for i in N for j in M) == 1 for f in Q)
mdl.addConstrs(quicksum(y[g, u, f]*(q[g, u]) for g in Q for u in U) + (2*b[f]) >= 1 for f in Q)
mdl.addConstrs(quicksum(y[g, u, f]*(q[g, u]) for g in Q for u in U) + (2*b[f]) <= 1 for f in Q)
mdl.addConstrs(quicksum(y[g, u, f] for f in Q for u in U) == 1 for g in Q)

#constraint for initial storage condition and rules
mdl.addConstr(quicksum(sc[i, j, u, 0] for i in N for j in M for u in U) == 0)
mdl.addConstrs(quicksum(sc[i, j, u, f] for u in U) <= 1 for i in N for j in M for f in Q)

#constraint for the task clustering 
mdl.addConstrs(quicksum(y[g+d, u, f+d] for d in A for u in U) - cl <= (1 - y[g, u, f]) * cl for g in S for f in C for u in U)
mdl.addConstrs(cl - quicksum(y[g+d, u, f+d] for d in A for u in U) <= (1 - y[g, u, f]) * cl for g in S for f in C for u in U)
mdl.addConstrs(y[g, u, f] == 0 for g in S for u in U for f in D)

#linearized version of xif and ygf for storage condition constraint
mdl.addConstrs(sc[i, j, u, f] - (sc[i, j, u, f-1] + q[g, u]*z[i, j, g, u, f]) <= 1 - z[i, j, g, u, f] for i in N for j in M for g in Q for u in U for f in Q)
mdl.addConstrs((sc[i, j, u, f-1] + q[g, u]*z[i, j, g, u, f]) - sc[i, j, u, f] <= 1 - z[i, j, g, u, f] for i in N for j in M for g in Q for u in U for f in Q)

mdl.addConstrs(sc[i, j, u, f] - sc[i, j, u, f-1] <= x[i, j, u, f] for i in N for j in M for u in U for f in Q)
mdl.addConstrs(sc[i, j, u, f-1] - sc[i, j, u, f] <= x[i, j, u, f] for i in N for j in M for u in U for f in Q)

mdl.addConstrs(quicksum(z[i, j, g, u, f] for i in N for j in M) - 1 <= 1 - y[g, u, f] for g in Q for f in Q for u in U)
mdl.addConstrs(1 - quicksum(z[i, j, g, u, f] for i in N for j in M) <= 1 - y[g, u, f] for g in Q for f in Q for u in U)

mdl.addConstrs(z[i, j, g, u, f] <= x[i, j, u, f] for i in N for j in M for f in Q for g in Q for u in U)
mdl.addConstrs(z[i, j, g, u, f] <= y[g, u, f] for i in N for j in M for f in Q for g in Q for u in U)
mdl.addConstrs(z[i, j, g, u, f] >= x[i, j, u, f] + y[g, u, f] - 1 for i in N for j in M for f in Q for g in Q for u in U)


#model solving
mdl.optimize() 
    
#model solution
mdl.printAttr('X') 

#debug and checking..
#sol = mdl.getAttr('X',mdl.getVars())

#for v in mdl.getVars(): 
#    print(v.x)

#infeasibility analyzer
#mdl.computeIIS()
#mdl.feasRelaxS(0,True,False,True)
