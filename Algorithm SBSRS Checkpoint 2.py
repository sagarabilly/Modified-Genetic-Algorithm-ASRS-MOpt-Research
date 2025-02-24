import random
import math
import time
from gurobipy import Model, GRB, quicksum 
import operator

#------------------------INITIALIZATION------------------------

#size variables
n = 10 # number of slots/locations
m = 3 # number of tiers
v = 3 # number of SKU
qq = 20 # the total of all demand tasks
cl = 3  # cluster of orders
tp = 1 #penalty in seconds

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

#set variables for helping decision variable
H = [(i, j, u, f) for i in N for j in M for f in Q for u in U]
F = [(g, u, f) for g in Q for u in U for f in Q]
G = [(i, j, u, f) for u in U for i in N for j in M for f in B]
P = [(i, j, g, u, f) for i in N for j in M for f in Q for g in Q for u in U]
T = [(f) for f in Q]

#time equations
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
countcost = len(c)

#for penalty cost 
for g in Q :
    for f in Q :
        if f > g :
            p[g,f] = abs(f-g)*tp
        if f <= g :
            p[g,f] = 0

#INPUT THE DATA SET
#demand of each task (storing or retrieving or do nothing)
q = {
     (1, 1): 0,(1, 2): 0,(1, 3): 1,
     (2, 1): 0,(2, 2): 0,(2, 3): 1,
     (3, 1): 0,(3, 2): 0,(3, 3): 1,
     (4, 1): 1,(4, 2): 0,(4, 3): 0,
     (5, 1): 0,(5, 2): 1,(5, 3): 0,
     (6, 1): 0,(6, 2): 1,(6, 3): 0,
     (7, 1): 0,(7, 2): 0,(7, 3): -1,
     (8, 1): 0,(8, 2): 0,(8, 3): -1,
     (9, 1): 0,(9, 2): -1,(9, 3): 0,
     (10, 1): 0,(10, 2): 0,(10, 3): 1,
     (11, 1): 1,(11, 2): 0,(11, 3): 0,
     (12, 1): 1,(12, 2): 0,(12, 3): 0,
     (13, 1): 0,(13, 2): 0,(13, 3): 1,
     (14, 1): 0,(14, 2): 1,(14, 3): 0,
     (15, 1): 0,(15, 2): 0,(15, 3): -1,
     (16, 1): 0,(16, 2): -1,(16, 3): 0,
     (17, 1): -1,(17, 2): 0,(17, 3): 0,
     (18, 1): 0,(18, 2): 0,(18, 3): -1,
     (19, 1): -1,(19, 2): 0,(19, 3): 0,
     (20, 1): 0,(20, 2): 0,(20, 3): -1
     } 
S = [1,4,7,12,15,18]
UNS = [10,11]

#Additional Statistics Process 
stot = len(S) #calculating total of S
unstot = len(UNS) #calculating total of UNS
spop = stot + unstot 
W = [0] 
hitungr = 0
#for calculating how mamny retrieval task in each SKU
for u in U :
    for g in Q:
        if q[g,u] == -1 :
            hitungr = hitungr + 1
    W.append(hitungr)
    hitungr = 0
modussku = max(W)
#multpr = int(qq/5) #for probability in priority dispatching rule (other start loc)
#comppr = int(qq/5) #for probability in priority dispatching rule (start at 0)

multpr = 2 #for probability in priority dispatching rule (other start loc)
comppr = 0 #for probability in priority dispatching rule (start at 0)

#--------------------------------------------------------------
#------------------THE ALGORITHM STARTS HERE-------------------
#______________________________________________________________

#GENETIC ALGORITHM PARAMETER
#ngenome = spop*24
ngenome = 150
looptime = 20
mutationprob = 1 #in percentage
crossoverprob = 90 #in percentage
elitismprob = 5 #in percentage
ngeneration = 200 #how many iteration that the algorithm will operate
#crossover param:
leftvalcompen = 0
rightvalcompen = 0
#minswathsize = math.ceil((spop/2) - 3) #a round up value of half of the numbers of orders
#minswathsize = math.ceil(spop/2) #a round up value of half of the numbers of orders
minswathsize = 1 #fixed value

comparison = 0 #iteration stop if this value is reach 
#_____________________________________________

# assign varables and lists 
z = {} #genome x
x = {} #genome y
y = {} #genome z
sc = {} #storage conditions in total coressponding the item type
scl = {} #storage conditions in each location for corresponding item type
itemcounterl = {} #additional variable act as a counter
lst = [] #lists of task that will be operated in sequence

# Initialize every decision variables equals to 0 (for fitness calculation purpose)
for f in Q:
    for u in U:
        for g in Q:
            for i in N:
                for j in M:
                    z[i,j,g,u,f] = 0
                    x[i,j,u,f] = 0
                    y[g,u,f] = 0

#assigning a populations of solutions
zz = [] # a population of z solutions
xx = [] # a population of x solutions
yy = [] # a population of y solutions
lstpop = [] # a population of lst solutions (not flatted version)
fitval = [] # for containing the fitness value 
swathchild1 = []

#assining another popul
zzz = []
xxx = []
yyy = []
lstpopul = []

#solution for preparation crossover
zzbestbv1 = []
zzbestbv2 = []
xxbestbv1 = []
xxbestbv2 = []
yybestbv1 = []
yybestbv2 = []
lstbestbv1 = []
lstbestbv2 = []

#solution for the crossover
zbestbv = {}
xbestbv = {}
ybestbv = {}
lstbestbv = []

#child solution for the crossover
zbestbvf1 = {}
xbestbvf1 = {}
ybestbvf1 = {}
lstbestbvf1 = []
zbestbvf2 = {}
xbestbvf2 = {}
ybestbvf2 = {}
lstbestbvf2 = []

#solution for after the crossover
zzbestov = []
xxbestov = []
yybestov = []
lstbestov = []

#solution for after mutation
zbest = {}
xbest = {}
ybest = {}
lstbest = []

#solution for after mutation
zbestt = []
xbestt = []
ybestt = []
lstbestt = []

#BEST LAST SOL
xtop = {}
ytop = {}
ztop = {}
lsttop = []

#-----------------------------DEFINE METHOD----------------------------

# obective function
def obj(x,y):
    return sum(x[i,j,u,f]*c[i,j] for i in N for j in M for f in Q for u in U) + sum(y[g,u,f]*p[g,f] for g in Q for f in Q for u in U)
    #return sum(x[i,j,u,f]*c[i,j] for i in N for j in M for f in Q for u in U) 

#fitness calculation
def fitness(x,y):
    ansobj = obj(x,y)
    
    #fitness calculation is based on the minimum value
    #fitness considered 100 if its near the value of total task*time required to store at location 1 tier 1 
    #which is impossible to achieve duh...
    if ansobj == qq*c[1,1]:
        return 100
    else :
        return (1/(abs(ansobj-(qq*c[1,1]))+1))*100
    #return (1/(abs(ansobj-(qq*c[1,1]))+c[1,1]))*100
    #add +c[1,1] for measurements and *100 so that it's in the form of percentage

#flattening list method
def flatten_list(x):
    for item in x:
        if type(item) in [list]:
            for num in flatten_list(item):
                yield(num)
        else:
            yield(item)

#checking the storage conditions (should be inside a loop correspoding with random shuffle -> do while)
def storage_conditions(lstflat):
    for i in range (0,qq-1) :
        g = lstflat[i]
        for u in U :
            #itemcounter = q[g,u]            
            #sc[u] = sc[u] + itemcounter
            sc[u] = sc[u] + q[g,u]
            if sc[u] < 0 :
                return False
    return True

#checking the storage conditions with location (should be inside a loop correspoding with random shuffle -> do while)
def storage_conditionslocation(g,i,j):
    for u in U :
        itemcounterl[u] = q[g,u]            
        scl[u,i,j] = scl[u,i,j] + itemcounterl[u] 
        if scl[u,i,j] < 0 : 
            scl[u,i,j] = scl[u,i,j] - itemcounterl[u] #RESET SCL IF FALSE
            return False
        if scl[u,i,j] > 1 :
            scl[u,i,j] = scl[u,i,j] - itemcounterl[u] #RESET SCL IF FALSE
            return False
        if scl[1,i,j] + scl[2,i,j] + scl[3,i,j] < 0 : # HAS TO BE SUM of ALL U (should be looping based on number of SKU)
            scl[u,i,j] = scl[u,i,j] - itemcounterl[u] #RESET SCL IF FALSE  
            return False        
        if scl[1,i,j] + scl[2,i,j] + scl[3,i,j] > 1 : # HAS TO BE SUM of ALL U (should be looping based on number of SKU)
            scl[u,i,j] = scl[u,i,j] - itemcounterl[u] #RESET SCL IF FALSE   
            return False
    return True

#getting the location i and j based on the closest distance 
def get_minvalue(inputdict,ind):
    keys_list = []
    keys_list.clear()
    #anstuple = min(inputdict, key=inputdict.get)
    #dict(sorted(inputdict.items(), key=lambda item: item[1])) #sorting the cij dictionary
    #key_list = sorted(inputdict.items(), key=operator.itemgetter(1))
    #key_list = {k: r for k, r in sorted(inputdict.items())}
    #keys_list = list(key_list)
    #keys_list = [*inputdict] #getting the keys of the cij dictionary
    sorted_cost = sorted(inputdict.items(),key=lambda item:item[1])
    for i in range (0,countcost) : 
        costcurtup = sorted_cost[i]
        keycostcur = costcurtup[0]
        keys_list.append(keycostcur)
    #print("the keylist : " , keys_list)
    keytup = keys_list[ind] #returning the keys based on the index
    #print("keylist : ", keys_list)
    #ind = ind + 1
    #print("this is the index calculation :" , ind)
    return keytup

#getting min value for crossover
def crsminval (lstcheck1,lstcheck2,crsmin,i):
    checkcrsmin = False
    crsmin = 0
    crsminnewval = crsmin + i
    while (checkcrsmin == False):
        if (crsminnewval > (int(ngenome/2)-1)):
            crsmin = 0
            crsminnewval = 0
            checkcrsmin = False
        if (lstcheck1 == lstcheck2[crsminnewval]) :
            #print("crsmin :", crsmin)
            #print("i : ", i)
            #print("crsminnewval : ", crsminnewval)
            crsmin = crsmin + 1
            crsminnewval = i + crsmin
            checkcrsmin = False
        else :  
            checkcrsmin = True
            #crsmin = crsmin + 1
    return crsmin+i

#Priority Based Rule (dispatching rule method)
def priority_rule(g):
    lind = 0
    for u in U : 
        if q[g,u] == 1 :
            prvalcomp = random.randint(1,100)
            prob = (W[u]/(modussku+comppr))*100
            if prvalcomp <= prob :
                lind = 0
            else :
                lind = random.randint(1,(v+multpr))
        if q[g,u] == -1 : #for retrieval, they always retrieve the shortest one
                lind = 0 
    return lind

#Random Generation Method        
def random_generation(numbco):
    #defining initial storage conditions 
    # Initialize everything equals to 0
    lst = []
    ind = 0
    for f in Q:
        for u in U:
            for g in Q:
                for i in N:
                    for j in M:
                        z[i,j,g,u,f] = 0
                        x[i,j,u,f] = 0
                        y[g,u,f] = 0                       
    for u in U :
        sc[u] = 0
        itemcounterl[u] = 0   
    for i in N :
        for j in M :
            for u in U :
                scl[u,i,j] = 0

    # create multiple list without a string name 
    for d in range (1,stot+1) :
        lst.append([S[d-1], S[d-1]+1, S[d-1]+2]) #there should be looping here for +0, +1, +2, etc.
    for d in range (unstot) :     
        rndpst = random.randint(0,stot)
        lstuns = UNS[d]
        lst.insert(rndpst, UNS[d])

    checkstc = False
    while (checkstc == False) :
        for u in U :
            sc[u] = 0
        random.shuffle(lst)    
        lstflat = list(flatten_list(lst)) #calling the method of flattening list
        checkstc = storage_conditions(lstflat) #calling the method of checking storage condition
        #print("looping process of checking storage condition")
       
    #genome random generation (method)
    #print(f"---these are random generated solutions:", numbco , "---")
    for f in Q :
        g = lstflat[f-1]        
        #u = random.randint(1,v)
        #i = 1 #if you want to assign anything in the first location
        #j = 1
        checkstcloc = False
        ind = 0
        ind = priority_rule(g) 
        while (checkstcloc == False) :
            anstup = get_minvalue(c,ind)
            i, j = anstup
            checkstcloc = storage_conditionslocation(g,i,j) #calling method for checking storage in each location
            #print("looping process of checking storage condition each location")
            #checkstcloc = True
            ind = ind + 1
        ind = 0
        
        for u in U :
            if abs(q[g,u]) == 1 :
                z[i,j,g,u,f] = 1
                x[i,j,u,f] = 1
                y[g,u,f] = 1
                #for printing only the solution (only when q[g,u] if equals to 1) -> only if there's demand
                #print("z[",i,",",j,",",g,",",u,",",f,"]")
                
            else :
                z[i,j,g,u,f] = 0
                x[i,j,u,f] = 0
                y[g,u,f] = 0     

    # for printing the result of the objective funtion
    objval = obj(x,y)
    #print("The objective function value is :", objval, "in seconds")   
    return z,x,y,lst

def generate_multiple_solutions(ngenome) :
    numbco = 1
    for i in range (0,ngenome) :
        zadd, xadd, yadd, lstadd = random_generation(numbco)
        #random_generation(numbco)
        #appending dictionary has to be translated into a copy first
        zaddcopy = zadd.copy()
        yaddcopy = yadd.copy()
        xaddcopy = xadd.copy()
        lstaddcopy = lstadd.copy()
        numbco = numbco + 1
        #appending each solutions to a population list
        zzz.append(zaddcopy)
        xxx.append(xaddcopy)
        yyy.append(yaddcopy)
        lstpopul.append(lstaddcopy)
    return zzz,xxx,yyy,lstpopul
    #return [random_generation() for _ in range (genpop)]
        
def selection_pair(zz,xx,yy,lstpop):
    fitval.clear() # Needs to be cleared
    for i in range (0,ngenome) :
        fitvalue = fitness(xx[i],yy[i])
        fitval.append(fitvalue)
    #dict(sorted(fitval.items(), key=lambda item: item[1])) #Command for sorting the dictionary ascending based on the fitness function value           
    
    #method 1 - sorting using zip
    zzsort = [zz for _, zz in sorted(zip(fitval, zz), key=lambda pair: pair[0])]
    xxsort = [xx for _, xx in sorted(zip(fitval, xx), key=lambda pair: pair[0])]
    yysort = [yy for _, yy in sorted(zip(fitval, yy), key=lambda pair: pair[0])]
    lstpopsort = [lstpop for _, lstpop in sorted(zip(fitval, lstpop), key=lambda pair: pair[0])]
    
    #Reversed the list
    zzsort.reverse()
    xxsort.reverse()
    yysort.reverse()
    lstpopsort.reverse()
    
    return zzsort,xxsort,yysort,lstpopsort 


def crossoverprocess(lstbestbvb1,lstbestbvb2):
    #for determinig the swath size at random
    #print(swathchild1)
    randomswathsize = random.randint(minswathsize,spop-1-rightvalcompen)
    #print("randswatssize :",randomswathsize)
    #left and right index at random
    hidari = random.randint(leftvalcompen,(spop-randomswathsize-rightvalcompen))
    #print("hidari :",hidari)
    migi = hidari + randomswathsize
    #print("migi : ",migi)
    
    #clearing the swathchild1
    swathchild1.clear()
    
    #appending values to all index of swathchild1 by 0
    for i in range (0,spop) : 
        swathchild1.append(0)
    
    ## this is for replacing the values on the corresponding swath (if needed)
    #for i in range (0,spop) :
    #    swathchild1[i] = 0 #reset/replacing all values to 0 (which means empty)
    
    for i in range (hidari,migi) :
        #swathchild1.insert(i, lstbestbv1[i]) #still dont know if the position is right or not #should be replace not insert
        swathchild1[i] = lstbestbvb1[i] #for replacing value
    
    #print("child value step 1 : ", swathchild1)      
    for i in range (hidari,migi) :
        checkval = lstbestbvb2[i] #checkval value never change
        checkvalnext = lstbestbvb2[i] #checkvalnext value is changing in every checking 
        if checkval not in swathchild1 : #if the value is not in the child then we have to add it
            checkvalnext = lstbestbvb1[i]
            #print("checkvalnext beg : ", checkvalnext)
            #finding the position of value checkval in lstbestbvb2
            postcheckval = lstbestbvb2.index(checkval)
            #print("postcheckval beg : ", postcheckval)
            while (swathchild1[postcheckval]) != 0 :  
                checkvalnext = lstbestbvb1[postcheckval]
                #finding the position of value checkval in lstbestbvb2
                postcheckval = lstbestbvb2.index(checkvalnext)
                #print ("looping check postcheckval :", postcheckval)
            swathchild1[postcheckval] = checkval
    
    #print("child value step 2 : ", swathchild1)             
    #drop all the value for the rest
    for i in range (0,spop) :
        if swathchild1[i] == 0 :
            swathchild1[i] = lstbestbvb2[i]
    
    swathchild1add = swathchild1.copy()
    #print("result for the child : ", swathchild1add)                
    return swathchild1add 

def crossover(lstbestbvb1,lstbestbvb2):
    # Initialize everything equals to 0
    lst = []
    ind = 0
    for f in Q:
        for u in U:
            for g in Q:
                for i in N:
                    for j in M:
                        zbestbv[i,j,g,u,f] = 0
                        xbestbv[i,j,u,f] = 0
                        ybestbv[g,u,f] = 0  
   
    for u in U :
        sc[u] = 0
        itemcounterl[u] = 0
    for i in N :
        for j in M :
            for u in U :
                scl[u,i,j] = 0
    
    #adding the values on the corresponding swath
    for i in range (0,spop) :
        swathchild1.append(0) #assigning all values to 0 (which means empty)
    
    crossoverloopc = 0
    checkstc = False
    while (checkstc == False and crossoverloopc < 30) :
        #initialize storage condition for checking
        for u in U :
            sc[u] = 0
        #calling crossover process method
        lstbestbv = crossoverprocess(lstbestbvb1,lstbestbvb2)
        #checking storage condition
        lstflatnewcr = list(flatten_list(lstbestbv)) #calling the method of flattening list
        checkstc = storage_conditions(lstflatnewcr) #calling the method of checking storage condition
        #print("looping process of checking storage condition")
        crossoverloopc = crossoverloopc + 1
        #print("crossoverloopc : ", crossoverloopc)
    
    if crossoverloopc == 30 :
        lstflatnewcr = list(flatten_list(lstbestbvb1))
        lstbestbv = lstbestbvb1
        #print("reaching limit of looping")
        
    #generating a new location assignment and checking storage condition locations
    #print(f"---Solutions after crossover:")
    for f in Q :
        g = lstflatnewcr[f-1]        
        #u = random.randint(1,v)
        #i = 1 #if you want to assign anything in the first location
        #j = 1
        checkstcloc = False
        ind = 0
        ind = priority_rule(g) 
        while (checkstcloc == False) :
            anstup = get_minvalue(c,ind)
            i, j = anstup
            checkstcloc = storage_conditionslocation(g,i,j) #calling method for checking storage in each location
            #print("looping process of checking storage condition each location")
            #checkstcloc = True
            #print("index number crossover : ", ind)
            ind = ind + 1
        ind = 0
        
        for u in U :
            if abs(q[g,u]) == 1 :
                zbestbv[i,j,g,u,f] = 1
                xbestbv[i,j,u,f] = 1
                ybestbv[g,u,f] = 1
                #for printing only the solution (only when q[g,u] if equals to 1) -> only if there's demand
                #print("z[",i,",",j,",",g,",",u,",",f,"]")
                
            else :
                zbestbv[i,j,g,u,f] = 0
                xbestbv[i,j,u,f] = 0
                ybestbv[g,u,f] = 0   
                
    objval = obj(xbestbv,ybestbv)
    #print("The objective function value after crossover :", objval, "in seconds")   
    return zbestbv,xbestbv,ybestbv,lstbestbv
    

def mutation(lstbestov):
    # Initialize everything equals to 0
    lst = []
    ind = 0
    for f in Q:
        for u in U:
            for g in Q:
                for i in N:
                    for j in M:
                        zbest[i,j,g,u,f] = 0
                        xbest[i,j,u,f] = 0
                        ybest[g,u,f] = 0                       
    for u in U :
        sc[u] = 0
        itemcounterl[u] = 0   
    for i in N :
        for j in M :
            for u in U :
                scl[u,i,j] = 0
                
    #The mutation program
    spoplist = [i for i in range (0,spop-1)]
    checkstc = False
    while (checkstc == False) :
        #initialization storage checking
        for u in U :
            sc[u] = 0
        #pick a 2 random position
        post1 = random.randint(0,spop-1)
        post2 = random.choice([el for el in spoplist if el != post1])
        #switch the value between the random position
        chromosome1 = lstbestov[post1]
        chromosome2 = lstbestov[post2]
        lstbestov[post2] = chromosome1
        lstbestov[post1] = chromosome2
        #checking storage condition
        lstflatnew = list(flatten_list(lstbestov)) #calling the method of flattening list
        checkstc = storage_conditions(lstflatnew) #calling the method of checking storage condition
        #print("looping process of checking storage condition")
    
    #generating a new location assignment and checking storage condition locations
    lstbest = lstbestov
    #print(f"---Solutions after mutation:")
    for f in Q :
        g = lstflatnew[f-1]        
        #u = random.randint(1,v)
        #i = 1 #if you want to assign anything in the first location
        #j = 1
        checkstcloc = False
        ind = 0
        ind = priority_rule(g) 
        while (checkstcloc == False) :
            anstup = get_minvalue(c,ind)
            i, j = anstup
            checkstcloc = storage_conditionslocation(g,i,j) #calling method for checking storage in each location
            #print("looping process of checking storage condition each location")
            #checkstcloc = True
            #print("index number mutation : ", ind)
            ind = ind + 1
        ind = 0
        
        for u in U :
            if abs(q[g,u]) == 1 :
                zbest[i,j,g,u,f] = 1
                xbest[i,j,u,f] = 1
                ybest[g,u,f] = 1
                #for printing only the solution (only when q[g,u] if equals to 1) -> only if there's demand
                #print("z[",i,",",j,",",g,",",u,",",f,"]")
                
            else :
                zbest[i,j,g,u,f] = 0
                xbest[i,j,u,f] = 0
                ybest[g,u,f] = 0   
    
    objval = obj(xbest,ybest)
    #print("The objective function value after mutation :", objval, "in seconds")   
    return zbest,xbest,ybest,lstbest


#----------------MAIN ALGORITHM PROCESS-------------------
#---------------------------------------------------------

#FIRST INITIAL SOLUTION
objinit = []
objinit.clear() 
zbestt,xbestt,ybestt,lstbestt = generate_multiple_solutions(ngenome) #generate initial ngenome solution
for i in range (0,ngenome) :
    objinit.append(obj(xbestt[i],ybestt[i]))

psolcur = min(objinit)
indexnewvalo = objinit.index(psolcur)
zbestc = zbestt[indexnewvalo]
xbestc = xbestt[indexnewvalo]
ybestc = ybestt[indexnewvalo]
lstbestc = lstbestt[indexnewvalo]

print("This is the INITIAL SOLUTION :")
for f in Q:
    for u in U:
        for g in Q:
            for i in N:
                for j in M:
                    if zbestc[i,j,g,u,f] == 1 :
                        print("z[",i,",",j,",",g,",",u,",",f,"]")
print("The objective function value for the Initial Solution is: ", psolcur) 

#setting up time and iteration
start_time = time.time()
elapsed_time = 0
itera = 0
psol = 1000000 #so that psol always lost at the first iteration
itlast = 0

#while (elapsed_time <= looptime and psol > comparison) : #looping based on time
while (itera < ngeneration and psol > comparison) : #looping based on number of generation
    #for calculating time 
    current_time = time.time()
    elapsed_time = current_time - start_time
    itera = itera + 1

    #clearing solution of population list
    zz.clear()
    xx.clear()
    yy.clear()
    lstpop.clear()

    ##appending dictionary has to be translated into a copy first
    #zbestapp = zbest.copy()
    #xbestapp = xbest.copy()
    #ybestapp = ybest.copy()
    #lstbestapp = lstbest.copy()     
    
    #zz,xx,yy,lstpop = generate_multiple_solutions(ngenome-1)
    ##solution append from last iteration
    #zz.append(zbestapp)
    #xx.append(xbestapp)
    #yy.append(ybestapp)
    #lstpop.append(lstbestapp)
    
    zz = zbestt.copy()
    yy = ybestt.copy()
    xx = xbestt.copy()
    lstpop = lstbestt.copy()
    zz,xx,yy,lstpop = selection_pair(zz,xx,yy,lstpop)
    
    #clearing the solution from before
    #zzbestbv1.clear()
    #xxbestbv1.clear()
    #yybestbv1.clear()
    #lstbestbv1.clear()
    #zzbestbv2.clear()
    #xxbestbv2.clear()
    #yybestbv2.clear()
    #lstbestbv2.clear()     
    
    #Crossover preparation
    hitungcrs = -1
    for i in range (0,int(ngenome/2)):
        #Taking the best solution 1 from crossover 
        hitungcrs = hitungcrs + 1
        zzbestbv1.append(zz[hitungcrs])
        xxbestbv1.append(xx[hitungcrs])
        yybestbv1.append(yy[hitungcrs])
        lstbestbv1.append(lstpop[hitungcrs])
        #Taking the best solution 2 from crossover
        hitungcrs = hitungcrs + 1
        zzbestbv2.append(zz[hitungcrs])
        xxbestbv2.append(xx[hitungcrs])
        yybestbv2.append(yy[hitungcrs])
        lstbestbv2.append(lstpop[hitungcrs])
    
    #clearing the list from the last iteration (preparing for the crossover)
    zzbestov.clear()
    xxbestov.clear()
    yybestov.clear()
    lstbestov.clear()
    
    #Add A copy for fixing the error
    zzbestbv1 = zzbestbv1.copy()
    xxbestbv1 = xxbestbv1.copy() 
    yybestbv1 = yybestbv1.copy()
    lstbestbv1 = lstbestbv1.copy()
    zzbestbv2 = zzbestbv2.copy()
    xxbestbv2 = xxbestbv2.copy()
    yybestbv2 = yybestbv2.copy()
    lstbestbv2 = lstbestbv2.copy()    
    
    #Crossover
    crsv1 = 0
    crsv2 = 0
    for i in range (0,int(ngenome/2)):
        docrossover = random.randint(1,100)
        #print("docrossover value:", docrossover)
        if docrossover <= crossoverprob :
            
            crsv1 = crsminval(lstbestbv1[i],lstbestbv2,crsv1,i) #to make sure the parent didnt crossover with similar parent
            #print ("crossover f1:", i , " x ", crsv1)
            zbestbvf1,xbestbvf1,ybestbvf1,lstbestbvf1 = crossover(lstbestbv1[i],lstbestbv2[crsv1])
            
            crsv2 = crsminval(lstbestbv2[i],lstbestbv1,crsv2,i) #to make sure the parent didnt crossover with similar parent
            #print ("crossover f2:", i , " x ", crsv2)
            zbestbvf2,xbestbvf2,ybestbvf2,lstbestbvf2 = crossover(lstbestbv2[i],lstbestbv1[crsv2])
            
            #taking the solution result from crossover to the mutation process
            #print ("lstbestbvf1 : ", lstbestbvf1)
            #print ("lstbestbvf2 : ", lstbestbvf2)
            zzbestov.append(zbestbvf1.copy()) #child 1
            zzbestov.append(zbestbvf2.copy()) #child 2
            xxbestov.append(xbestbvf1.copy()) #child 1
            xxbestov.append(xbestbvf2.copy()) #child 2
            yybestov.append(ybestbvf1.copy()) #child 1
            yybestov.append(ybestbvf2.copy()) #child 2
            lstbestov.append(lstbestbvf1.copy()) #child 1
            lstbestov.append(lstbestbvf2.copy()) #child 2

        elif docrossover > crossoverprob : 
            #Taking the best soluton for mutation before crossover
            #crsv1 = crsv1 +1 #to make sure the value is on track even though not conducting crossover
            #crsv2 = crsv2 +1 #to make sure the value is on track even though not conducting crossover
            zzbestov.append(zzbestbv1[i]) #child 1
            zzbestov.append(zzbestbv2[i]) #child 2
            xxbestov.append(xxbestbv1[i]) #child 1
            xxbestov.append(xxbestbv2[i]) #child 2
            yybestov.append(yybestbv1[i]) #child 1
            yybestov.append(yybestbv2[i]) #child 2
            lstbestov.append(lstbestbv1[i]) #child 1
            lstbestov.append(lstbestbv2[i]) #child 2
            
    #Mutation clear prepare
    zbestt.clear()
    ybestt.clear()
    xbestt.clear()
    lstbestt.clear()
    
    for i in range (0,ngenome):        
        domutation = random.randint(1,100)
        #print("domutation value:", domutation)
        if domutation <= mutationprob :
            zbest,xbest,ybest,lstbest = mutation(lstbestov[i])
            zbestt.append(zbest)
            xbestt.append(xbest)
            ybestt.append(ybest)
            lstbestt.append(lstbest)
        elif domutation > mutationprob : 
            #taking the best solution before mutation
            zbestt.append(zzbestov[i])
            xbestt.append(xxbestov[i])
            ybestt.append(yybestov[i])
            lstbestt.append(lstbestov[i])
        
    #Elitism
    for i in range (0,ngenome): 
        doelitism = random.randint(1,100) 
        #print("doelitism value:", doelitism)
        if doelitism <= elitismprob :
            if obj(xbestt[i],ybestt[i]) > obj(xx[i],yy[i]):
                #print("The objective function value for best: ", obj(xbest,ybest))
                #print("The objective function value for parents: ", obj(xx[0],yy[0]))
                zbestt[i] = zz[i] 
                xbestt[i] = xx[i]
                ybestt[i] = yy[i]
                lstbestt[i] = lstpop[i]
                #print("previous solutions are better")
            #else :
                #print("new solutions are better")
                #print("The objective function value for best: ", obj(xbest,ybest))
                #print("The objective function value for parents: ", obj(xx[0],yy[0]))
    
    #Objit is a list of all objective function value of candidate populations
    objit = []
    objit.clear()
    for i in range (0,ngenome) :
        objit.append(obj(xbestt[i],ybestt[i]))
    objitm = min(objit)    
    #print("==============================================")
    print("ITERATION : ", itera , " OBJ VAL = ", objitm, " Time : ", elapsed_time)
    #print("==============================================")
    
    #for updating best sol and best time
    if objitm < psol :
        besttime = elapsed_time
        itlast = itera
        psol = objitm
        #THIS IS FINAL BEST RESULT OF ALL ITERATION
        indexnewval = objit.index(objitm) #finding the index of the best solution from the population
        xtop = xbestt[indexnewval]
        ytop = ybestt[indexnewval]
        ztop = zbestt[indexnewval]
        lsttop = lstbestt[indexnewval]

print("This is the BEST SOLUTION :")
for f in Q:
    for u in U:
        for g in Q:
            for i in N:
                for j in M:
                    if ztop[i,j,g,u,f] == 1 :
                        print("z[",i,",",j,",",g,",",u,",",f,"]")
print("The objective function value for the BEST SOLUTION is: ", psol) 
print("The time for producing that solution is : ", besttime)
print("Iteration for producing that solution is : ", itlast)

objlast = []
objlast.clear()
for i in range (0,ngenome):
    objlast.append(obj(xbestt[i],ybestt[i]))
#print("This is the LAST SOLUTION :")
#for f in Q:
#    for u in U:
#        for g in Q:
#            for i in N:
#                for j in M:
#                    if zbest[i,j,g,u,f] == 1 :
#                        print("z[",i,",",j,",",g,",",u,",",f,"]")
print("---------------------------------------------------------------")
print("The objective function value for the LAST SOLUTION is: ", min(objlast)) 
print("The elapsed time is : ", elapsed_time)
