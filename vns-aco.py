import random as rd
import numpy as np
import math
from numpy import inf
import time

rd.seed(141)

#given values for the problems
bins = []
best_route = []
b_status = np.array([])
d1 = np.array([])
rute = np.array([])
rute_dp = np.array([])
rute_d1 = np.array([])
rute_d2 = np.array([])
T_path = np.array([])
Actual_path = np.array([])
m_cost = np.array([])
m_cost = np.ones(3)

d = np.array([[1000,6.5,8.2,4.7,1.4,3.4,0.6,3.5,3.9,11.5,9.1,5.2,2.3,2.6,3.4],
             [6.5,1000,1.7,2.1,6.1,3.2,7.1,3.6,2.9,5.1,2.6,1.4,5.9,4.1,3.5],
             [8.2,1.7,1000,3.6,7.7,4.9,8.7,5.2,4.5,3.5,1.1,3.1,7.5,3.9,3.5],
             [4.7,2.1,3.6,1000,4.1,1.9,5.3,2.7,2.2,7.2,4.7,1.6,3.8,2.1,1.4],
             [1.4,6.1,7.7,4.1,1000,3.3,1.7,3.9,3.7,11.1,8.7,4.9,0.9,1.9,2.6],
             [3.4,3.2,4.9,1.9,3.3,1000,3.9,0.7,0.6,8.2,5.8,1.8,3.5,1.5,1.8],
             [0.6,7.1,8.7,5.3,1.7,3.9,1000,4.1,4.5,12.1,9.7,5.7,2.7,3.1,4.1],
             [3.5,3.6,5.2,2.7,3.9,0.7,4.1,1000,0.6,8.2,6.1,2.1,4.1,2.2,2.6],
             [3.9,2.9,4.5,2.2,3.7,0.6,4.5,0.6,1000,7.7,5.4,1.5,4.2,2.2,2.3],
             [11.5,5.1,3.5,7.2,11.1,8.2,12.1,8.2,7.7,1000,2.4,6.3,10.9,9.1,8.6],
             [9.1,2.6,1.1,4.7,8.7,5.8,9.7,6.1,5.4,2.4,1000,3.9,8.4,6.7,6.1],
             [5.2,1.4,3.1,1.6,4.9,1.8,5.7,2.1,1.5,6.3,3.9,1000,4.9,2.9,2.6],
             [2.3,5.9,7.5,3.8,0.9,3.5,2.7,4.1,4.2,10.9,8.4,4.9,1000,2.1,2.3],
             [2.6,4.1,3.9,2.1,1.9,1.5,3.1,2.2,2.2,9.1,6.7,2.9,2.1,1000,0.9],
             [3.4,3.5,3.5,1.4,2.6,1.8,4.1,2.6,2.3,8.6,6.1,2.6,2.3,0.9,1000]])


### DEPOT to bin
### DISPOSAL 1 to bin
### DISPOSAL 2 to bin
B = np.array([[4.8,2.5,3.9,2.9,4.6,2.1,5.9,2.1,1.4,6.1,4.0,0.8,4.7,2.9,2.8],
       [9.8,4.2,3.6,6.6,9.2,6.9,10.2,6.6,6.4,2.2,2.5,5.4,9.7,7.7,7.6],
       [2.1,7.1,8.7,5.9,0.9,3.9,2.1,4.6,4.7,11.8,9.5,5.9,1.1,2.5,3.6]])


iteration = 10
n_ants = 15
n_citys = 15
v_load = 1000

d1_dp = 6.4  ####### disposal1 to depot distance
d2_dp = 7.2  ####### disposal2 to depot distance

#######################  Initial bin allocation cost for each bin types
cost_b1 = 8.0
cost_b2 = 8.0
cost_b3 = 10.0

##################################### Bin service time ##############################

srt = 10

#################### For penalty cost ###############
p_t = 1.65  #### penalty limit
a = 10
b = 12

############################# for driver salary per hour ##

d_slr = 20   ### 20 dollar per day
number_of_driver = 2

########## per unit Routing Cost for different vehicle ###########################
v_1,v_2,v_3 = 1.25,1.25,1.25
v_4 = 1.45
v_5 = 1.55

###### Vehicle average velocity ##############
vav = 40    ### km/hr









#Starting time
start=time.time()
# intialization part
m = n_ants
n1 = n_citys
e = .5         #evaporation rate
alpha = 1     #pheromone factor
beta = 2       #visibility factor
b_status = np.ones(n1)

####  calculating the load status of all bins.
def collect_information(bins,b_status):
    
    
    #print(b_status)
    print("Filling status of all bins: ")
    for i in range(n1):                   ### initialize the load of all bins
        b_status[i] = rd.randint(60,100) 
    print(b_status)
    #bins = []
    global b1     ### how many bins are filled
    b1 = 0
    i1 = 0
    for i in range(n1):
         #if ((b_status[i]<72) or (b_status[i]>95)):
          #if (b_status[i]>=80):
            
           if (b_status[i]<80):
             bins.append(i)
             b1 += 1
             i1 += 1
           if (b_status[i]>=80):
               bins.append(i)
               b1 += 1
               i1 += 1
    
    #bins = np.array(bins)
    print("How many bins fill >= 90% ?  ",b1)
    print(bins)
    return bins,b_status
#End time
#end=time.time()
#Total time taken
#print(f"Runtime of the program is {end-start}")
#visibility = 1/d
#visibility[visibility == inf ] = 0
def vns_path():
    global s_b
    s_b = 0        ####  s_b = starting bin
    
    collect_information(bins,b_status)
    temp = bins.copy()
    
    print(b1)
    bins_d = []
    b = 0
    for i in range(b1):
        b = int(temp[i])
        c = B[0][b]
        bins_d.append(c)
    print(bins_d)
    m_d = np.argmin(bins_d)   #### m_d = min distance from depot
    global S
    S = m_d
    print("S =",S)
    rute = bins[m_d]
    print("RUTE = ",rute)


    ACO()
    #find_best_path()
    chk_load()
    
    
    
    
    
    
def ACO():
    iteration = 1
    n_ants = 15
    n_citys = b1

# intialization part

    m = n_ants
    global n
    n = n_citys
    e = .5         #evaporation rate
    alpha = 1     #pheromone factor
    beta = 2       #visibility factor
    
    
    
    d1 = np.ones((n,n))
    #T_path = np.zeros((3,b1))
    #print(d1)
    k,k1 = 0,0
    #for k in range(n):
    for i in range(n):
        k1 = 0
        for j in range(n):
             d1[i][j] = d[int(bins[k]),int(bins[k1])]
             #print("I am here NOW")
             k1 += 1
        k += 1
    print(d1)
    #d1[d1 == 1000 ] = 0
    #print(d1)

    

#calculating the visibility of the next city visibility(i,j)=1/d(i,j)

    visibility = 1/d1
    visibility[visibility == inf ] = 1000

#intializing pheromne present at the paths to the cities

    pheromne = .1*np.ones((m,n))

#intializing the rute of the ants with size rute(n_ants,n_citys+1) 
#note adding 1 because we want to come back to the source city

    rute = np.ones((m,n))

    for ite in range(iteration):
    
        rute[:,0] = 1          #initial starting and ending positon of every ants '1' i.e city '1'
    
        for i in range(m):
        
            temp_visibility = np.array(visibility)         #creating a copy of visibility
        
            for j in range(n-1):
                #print("Iterasi ke-" + str(ite+1) + " (Ant ke-" + str(i+1) + ") (Rute ke-" + str(j+1) + ")")
                #print("Rute awal: " + str(rute))
                combine_feature = np.zeros(5)     #intializing combine_feature array to zero
                cum_prob = np.zeros(5)            #intializing cummulative probability array to zeros
                #print("cum_prob init: " + str(cum_prob))
                #print("cur_loc: " + str(int(rute[i,j])))
                cur_loc = int(rute[i,j]-1)        #current city of the ant
                temp_visibility[:,cur_loc] = 0     #making visibility of the current city as zero
            
                p_feature = np.power(pheromne[cur_loc,:],beta)         #calculating pheromne feature 
                v_feature = np.power(temp_visibility[cur_loc,:],alpha)  #calculating visibility feature
            
                p_feature = p_feature[:,np.newaxis]                     #adding axis to make a size[5,1]
                v_feature = v_feature[:,np.newaxis]                     #adding axis to make a size[5,1]
            
                combine_feature = np.multiply(p_feature,v_feature)     #calculating the combine feature
                        
                total = np.sum(combine_feature)                        #sum of all the feature
            
                probs = combine_feature/total   #finding probability of element probs(i) = comine_feature(i)/total
            
                cum_prob = np.cumsum(probs)     #calculating cummulative sum
                #print('cum_prob: ' + str(cum_prob))
                r = np.random.random_sample()   #randon no in [0,1)
                #print('random r: ' + str(r))
                #print('Nilai (cum_prob>r): ' + str(cum_prob>r))
                #print('nonzero all: ' + str(np.nonzero(cum_prob>r)))
                #print('nonzero[0][0]: ' + str((np.nonzero(cum_prob>r)[0][0])))
                #print('(nonzero[0][0])+1: ' + str((np.nonzero(cum_prob>r)[0][0])+1))
                city = (np.nonzero(cum_prob>r)[0][0])+1       #finding the next city having probability higher then random(r) 
                #print('Next city: ' + str(city))
                #print('\n')
            
                rute[i,j+1] = city              #adding city to route 
           
            left = list(set([i for i in range(1,n+1)])-set(rute[i,:-2]))[0]     #finding the last untraversed city to route
        
            rute[i,-2] = left                   #adding untraversed city to route
       
        rute_opt = np.array(rute)               #intializing optimal route
    
        dist_cost = np.zeros((m,1))             #intializing total_distance_of_tour with zero 
    
        for i in range(m):
        
            s = 0
            for j in range(n-1):
            
                s = s + d1[int(rute_opt[i,j]-1),int(rute_opt[i,j+1]-1)]   #calcualting total tour distance
        
            dist_cost[i]=s                      #storing distance of tour for 'i'th ant at location 'i' 
       
        dist_min_loc = np.argmin(dist_cost)             #finding location of minimum of dist_cost
        dist_min_cost = dist_cost[dist_min_loc]         #finging min of dist_cost
    
        best_route = rute[dist_min_loc,:]               #intializing current traversed as best route
        pheromne = (1-e)*pheromne                       #evaporation of pheromne with (1-e)
    
        for i in range(m):
            for j in range(n-1):
                dt = 1/dist_cost[i]
                pheromne[int(rute_opt[i,j])-1,int(rute_opt[i,j+1])-1] = pheromne[int(rute_opt[i,j])-1,int(rute_opt[i,j+1])-1] + dt   
            #updating the pheromne with delta_distance
            #delta_distance will be more with min_dist i.e adding more weight to that route  peromne

    #print('route of all the ants at the end :')
    #print(rute_opt)
    #print()
    print('best path :',best_route)
    f_rute = []
    for i in range(b1):
        a1 = bins[int(best_route[i])-1]
        f_rute.append(a1)
    print('final path :',f_rute)
###print(dist_min_cost[0])
    print('cost of the best path',int(dist_min_cost[0]) )
    #+ d1[int(best_route[-2])-1,0]
    m_cost[0] = dist_min_cost[0]
    
    
############################################# NSA algo : search from deopt to Disposal center  ##########################################    
#def optimize(best_route,d1):
    
    r_d = np.array([])
    r_d = np.ones((n))
    rute_dp = np.ones((n))
    x = np.array([])
    temp1 = np.arange(b1)
    temp = []
    temp = best_route.copy()
    temp = temp - 1
    
    #print("Before temp =",temp)
    #print("rute_dp =",rute_dp)
    s1 = S.copy()
    s1 -=1
    #print("s1 = ",s1)
    i = 0
    k = 0
    
    a = np.where(temp == s1)
   
    temp[i],temp[a] = temp[a],temp[i]
    temp1[temp1 == temp[i]] = 9999
    x[x == temp[i]] = 9999
    #print("After temp =",temp)
    
    x = np.zeros((n))    
    for i in range(0,n-1):
        for j in range((n)):
            if(temp1[j] != 9999 ):
                x[j] = d1[int(temp[i]),j]
                #print("temp[i] = ",temp[i])
                #print("x[j] = ",x[j])
            else:
                x[j] = 9999
            
        #print("x_ = ",x)
        x_loc = np.argmin(x)
        #print("x_loc = ",x_loc)
        s1 = x_loc
        
        a = np.where(temp == s1)
        i += 1
        #print("a = ",a)
        temp[i],temp[a] = temp[a],temp[i]
        temp1[temp1 == temp[i]] = 9999
        x[x == temp[i]] = 9999
        
        
        
        #print("temp =",temp)
        rute_dp[:] = temp
        #print("rute_dp =",rute_dp)
    s2 = 0
    for j1 in range(n-1):
        s2 = s2 + d1[int(temp[j1]),int(temp[j1+1])]   #calcualting total tour distance
        
    #print("Now J =",s2)
    m_cost[1] = s2
    print()
    print()
        
############################################ NSA algo:  search from D1 to depot #######################
    temp1 = np.arange(b1)
    #print("temp1 =",temp1)
    temp2 = []
    temp2 = best_route.copy()
    temp2 = temp2 - 1
    
    r_d = np.ones((n))
    rute_d1 = np.ones((n))
    d_d1 = []
    
    #print("d_d1 =",d_d1)
    
    b = 0
    for i in range(b1):
        b = int(temp2[i])
        c = B[1][b]
        d_d1.append(c)
    print(d_d1)
    m_d = np.argmin(d_d1)   #### m_d = min distance from depot    
    s1 = m_d
    
    #print("Before temp2 =",temp2)
    #print("rute_d1 =",rute_d1)
    
    
    #print("s1 = ",s1)
    i = 0
    k = 0
    
    a = np.where(temp2 == s1)
    
    temp2[i],temp2[a] = temp2[a],temp2[i]
    temp1[temp1 == temp2[i]] = 9999
    x[x == temp2[i]] = 9999
    #print("After temp =",temp2)
    
    x = np.zeros((n))    
    for i in range(0,n-1):
        for j in range((n)):
            if(temp1[j] != 9999 ):
                x[j] = d1[int(temp2[i]),j]
                #print("temp2[i] = ",temp2[i])
                #print("x[j] = ",x[j])
            else:
                x[j] = 9999
            
        #print("x_ = ",x)
        x_loc = np.argmin(x)
        #print("x_loc = ",x_loc)
        s1 = x_loc
        
        a = np.where(temp2 == s1)
        i += 1
        #print("a = ",a)
        temp2[i],temp2[a] = temp2[a],temp2[i]
        temp1[temp1 == temp[i]] = 9999
        x[x == temp2[i]] = 9999
        
        
        
        #print("temp2 =",temp2)
        #print("temp1 =",temp1)
        rute_d1[:] = temp2
        res = temp2[::-1]
        #print("Reverse_temp2 =",res)
    s2 = 0
    for j1 in range(n-1):
        s2 = s2 + d1[int(res[j1]),int(res[j1+1])]   #calcualting total tour distance
        
    #print("Now J1 =",s2)
    m_cost[2] = s2

##############################################  find_best_path():  ############################
#def find_best_path():
    Actual_path = np.ones((b1))
    T_path = np.ones((b1))
    print("Initial T_path = ",T_path)
    print("Initial costs of three results = ",m_cost)
    x = np.argmin(m_cost)
    if(x == 0):
       T_path = best_route.copy() 
    print(x)
    if(x == 1):
       T_path = rute_dp.copy()+1
    if(x == 2):
       T_path = rute_d1.copy()+1
       
    print("Final selected route is =",T_path)
    for i in range(b1):
        a1 = bins[int(T_path[i])-1]
        Actual_path[i] = a1    
    print("Actual selected path is =",Actual_path)
    print(b_status)
    
############  calculate total cost: routing +  disposal to depot + bin allocation cost per day ####
    bb = Actual_path[-1]
    print("bb = ",bb)
    bb = int(bb)
    bb1 = B[1][bb]
    bb2 = B[2][bb]
    if(bb1<bb2):
        bb3 = bb1
        bb3 = d1_dp
    else:
        bb3 = bb2
        bb3 = d2_dp
    r_cost = (int(m_cost[x]) * v_1) + (bb3 * v_5) 
    print("Accumulataed cost = ",r_cost)
    
    
##################################################################################################
def chk_load():
    
    b2,b3 = [],[]
    b2,b3 = bins,b_status
    global total_load
    total_load = 0
    for i in range(b1):
        a = b2[i]
        total_load += b3[a]
    
    print("Total load = ",total_load)
vns_path()

while (total_load > v_load):
   vns_path()
   
end=time.time()
#Total time taken
print(f"Runtime of the program is {end-start}")

def penalty():
    x = np.argmin(m_cost)
    print("X= ",x)
    t1 = int(m_cost[x])/vav
    print("t1= ",t1)
    t2 = (b1-1) * srt
    t2 = t2 / 60
    print("t2= ",t2)
    t = t1 + t2
    print("Total time = ",t)
    
################ check the penalty ############################
    if(t<p_t):
        print("Total penalty is Nil")
    else:
        p_t1 = t - p_t
        p_t1 = (a + (b * p_t1))
        print("Total penalty = ",p_t1)
        
    
penalty()    