import math as m
from math import radians as rad, degrees as deg
import time
import random
from sys import stderr
import numpy as np
import copy

simulation = False
try:
    import simulator
    from operator import attrgetter
    simulation = True
    simulator.InitializePods()
except:
    pass

physics_evolv_time = 0.022
physics_evolv_time_again = 0.010
iteration_limit_for_sim = 25
future_moves_to_sim = 6
max_chromes_to_keep = 3
start_pop_random_chromes = 3
my_shield_chances = 0.2
turn_gain = 2
thrust_gain = 1.5
global_mutation_rate = 0.25
global_odds_to_mutate = 1
max_combined_vel_for_enemy_shield = 800

cps = {}
pods = {}
num_of_pods = 4
db = stderr #shortcut for debugging
max_t = 200
checkpoint_radius = 600
pod_radius = 400
StillPlaying = True
racer_pod, attacker_pod, prey_pod, defender_pod = None, None, None, None
direct_bot_states = {}
dist_sq_dict = {}
bio_div_log = []

#====================================================
#===========Neural Network Constants=================
input_layer_size = 20
hidden_layer_sizes = [8,8,8]
output_layer_size = 2
activation_func = 'sigmoid'
Lambda = 9e-6
weights = [np.array([[-8.19058541e-01, -6.54742261e-02,  1.59750598e+00,  1.30141020e+00,  3.08913119e+00, -3.70626214e+00,  8.14024052e-01, -1.09223039e-01],
       [ 1.55355928e+00, -9.80495039e-01,  1.38484560e+00, -7.60106342e-01, -2.47430220e-01, -3.42536339e+00,  3.64136877e-01,  3.37522425e+00],
       [-5.32859390e-02, -3.17003537e+00, -4.93571054e+00,  8.82025777e-02, -1.25448190e+00,  5.93313754e+00,  6.35823096e-02, -1.03009361e+00],
       [-2.05261402e+00,  3.72075680e+00, -2.09916538e+00,  2.72320765e+00, -3.90279024e+00, -1.78309603e+00,  8.94578124e+00,  1.71250427e+00],
       [ 9.87049804e+00,  4.84178509e-01, -1.99664015e-01,  2.18036068e+00, -4.39427591e-01, -1.55279833e+00,  1.78046950e+00,  2.86625299e+00],
       [-8.03798854e-01, -3.83567214e+00, -9.88820805e-01, -5.52313417e+00,  2.23521032e+00, -1.16264672e+00, -1.53924296e+00, -2.69063901e-01],
       [ 7.55743042e-01, -1.31204480e+00, -1.25616424e+00, -1.11539036e+00, -9.75509650e-01, -3.17236306e+00,  2.58647671e-01,  5.77280447e-01],
       [ 4.70589211e-01, -2.26394941e-01,  5.89174694e-01,  2.12840013e-01,  8.89391411e-01, -1.74804099e+00,  1.19260528e-01,  1.14929708e+00],
       [ 2.67817429e+00,  4.03854583e-01, -2.37975532e+00, -2.54647266e+00, -3.50033279e+00,  2.84213838e+00, -4.58224635e-01,  1.66680439e-01],
       [-7.60264849e-02, -1.57543902e-02,  4.41454744e-01, -1.31797954e-01,  1.09935132e+00, -7.35302658e-03, -2.26257988e-01, -2.77430009e-01],
       [ 3.28380907e-01, -8.71529271e-01, -9.37777828e-01,  1.00873405e+00, -9.77255165e-02,  1.64220640e-01,  4.70327400e-01,  1.03519889e+00],
       [-2.25725536e+00, -4.31338942e+00, -5.35619815e+00,  1.74321438e+00, -5.50476920e+00, -3.39471462e+00,  2.75882096e+00,  3.08046790e+00],
       [-4.36656068e-01, -5.12614287e-01, -4.48572299e-01,  1.84341580e+00,  1.23916256e-01,  1.42435236e+00,  7.58740812e-01,  2.19182834e-01],
       [-4.25585393e-01, -7.17406059e-01, -1.09820333e+00, -5.21185265e-01,  9.08880983e-02, -3.34718025e+00,  7.67180493e-01,  6.19268242e-01],
       [-2.39722772e+00,  1.56328109e+00,  1.57711087e+00,  2.10231776e+00,  2.71399790e-02, -8.88807746e-01, -1.48414933e-01, -1.21720369e+00],
       [ 1.08987383e-01,  1.27049341e-01,  1.08835746e+00,  1.01270504e-01,  1.15249484e+00,  3.55076637e-01, -1.39141060e-01, -2.19945996e-01],
       [-4.94455472e-01,  1.25400263e-01,  6.04015247e-01, -5.18236583e-01,  1.26113592e+00,  2.03959710e-01, -3.16334981e-01, -8.98144937e-01],
       [-7.78923517e-01, -2.35907806e+00, -2.67345520e+00, -4.31943457e+00,  3.61086280e+00, -2.51832000e-01, -2.28360383e+00, -4.09218822e+00],
       [-5.83659499e-02,  1.60149415e+00,  2.56821372e-01,  1.52309203e+00, -8.38523050e-01,  2.64027408e-02, -4.18060936e-01, -6.66700002e-01],
       [ 4.58489237e-01,  4.61279231e-01,  2.83979635e+00, -1.58900398e+00,  1.90006786e+00, -2.93143082e+00, -8.38248657e-01,  4.09545519e-01]]), np.array([[-1.09118146e+00, -4.93556893e+00,  6.71120929e+00, -3.67607235e+00, -1.61820035e+00,  1.02786702e+00, -2.13184379e+00, -2.19006263e+00],
       [ 4.13055408e-01, -3.25539558e+00, -1.23699847e+00,  4.72560072e-01, -1.75418432e+00,  1.46182291e+00, -9.11948647e-01,  6.21587475e+00],
       [-5.18955335e-01,  1.59059119e+00, -2.15189508e+00, -2.02936846e-01, -3.93824663e+00, -2.64284867e+00,  4.12100576e+00, -4.36096542e+00],
       [ 8.52551237e-01,  4.94563097e+00, -3.78935434e+00,  1.52412626e+00,  1.26873048e+00,  3.46752601e+00, -1.14348842e+00, -5.40064540e+00],
       [-2.57431012e-01, -4.67553439e-01, -3.29372444e-02, -6.89019636e-01,  3.08721133e+00,  2.70584061e-02, -2.38308842e+00,  3.71134804e+00],
       [-8.81342580e-01, -2.94686999e-02, -3.93156108e+00,  3.92562103e+00,  2.43509449e+00,  1.71443067e+00,  1.35917518e+00,  6.20615384e+00],
       [-7.34461377e+00, -3.57097159e+00, -5.36097299e-03,  3.64395597e+00, -7.07943379e-02,  2.15229577e-01, -3.95967091e-01, -1.39033459e-01],
       [ 2.94934636e+00,  7.66835132e-01, -2.39825050e+00,  1.63297854e+00, -1.44883523e+00,  1.33454679e-01,  3.73975385e+00,  5.27589198e+00]]), np.array([[-5.07989001e-01, -1.43439044e-01,  2.79796578e+00,  2.10351374e+00, -3.73131444e+00, -3.03613153e-01, -3.41817491e+00, -9.03417201e-01],
       [ 3.14148062e+00, -1.27500905e+00,  3.28507657e+00, -5.24027497e+00, -1.25491778e+00,  3.77100128e-01,  2.09562718e+00,  8.08399500e-01],
       [ 1.51451153e+00,  2.12505193e+00, -9.49212338e-01, -1.60931217e+00, -2.63829359e+00, -5.61114709e-01,  6.30348821e+00,  1.61015971e-01],
       [-2.42446140e+00,  1.63164410e-01, -1.53256027e+00,  5.98715402e+00, -1.19681711e+00, -3.16528394e-01, -1.03039217e+00, -2.85191177e-01],
       [-1.63168288e+00,  8.11383935e-01,  5.38763874e-01,  3.07985504e+00,  2.02097961e+00, -1.11822609e+00, -5.59410094e-01, -3.99825049e-01],
       [-3.38314710e-02, -1.41347981e-01, -2.15637694e+00, -4.60628560e+00,  1.02754375e+00,  1.43494115e-01,  9.83733295e-01, -3.19889056e-01],
       [ 1.41889604e+00,  2.68077758e+00, -1.14259091e+00,  3.65833843e+00, -9.05397010e-03,  1.28065952e+00,  3.31748767e+00,  1.06439563e+00],
       [ 2.27564368e+00, -3.89670559e+00, -1.82155922e+00, -2.02546895e+00,  3.92030470e+00, -1.27143146e+00, -1.06165221e+01, -1.02012775e+00]]), np.array([[-4.96289151,  0.42006655],
       [ 0.6633981 , -5.48207348],
       [ 0.79539563, -4.65513682],
       [ 8.30339552,  2.81987412],
       [-4.75130233,  0.34719525],
       [-0.9141426 , -1.85460366],
       [-2.19119928,  7.93683534],
       [-1.76916158, -0.88652708]])]
bias_weights = [
            np.array([[ 1.27164291,  3.12095976,  2.05902641,  2.57663801, -0.28510143,  1.17991576, -5.01127067, -2.69593035]]),
            np.array([[ 0.05729189,  2.76790881, -0.27532979,  2.02718541,  1.76093016, -2.7869614 , -0.24081686, -2.39885238]]),
            np.array([[ 0.24244545,  0.89410648, -0.95857375, -0.41821948, -0.05544261,  0.12174627,  0.30937812,  0.19170227]]),
            np.array([[-0.27268567, -0.75968183]])
            ]
input_normalization = np.array([[ 0.00000000e+00, -9.44000000e+02, -9.63000000e+02, -3.14159265e+00,  0.00000000e+00, -3.14159265e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -3.14159265e+00,  0.00000000e+00, -1.50700000e+03, -1.28900000e+03,  0.00000000e+00,  0.00000000e+00, -3.14159265e+00,  0.00000000e+00, -1.66800000e+03, -1.47100000e+03],
       [ 1.00000000e+00,  1.00000000e+03,  8.97000000e+02,  3.14159265e+00,  2.50000000e+04,  3.14159265e+00,  2.50000000e+04,  1.00000000e+00,  1.00000000e+00,  6.28318531e+00,  3.14159265e+00,  2.50000000e+04,  1.48200000e+03,  1.46900000e+03,  1.00000000e+00,  6.28318531e+00,  3.14159265e+00,  2.50000000e+04,  1.96400000e+03,  1.42300000e+03]])

def SetCommands(p, op):
    raw_gene = p.best_chrome.chrome[0]
    act_gene = ConvertGenToActual(raw_gene)
    turn, thrust = act_gene[0], act_gene[1]
    shield = 1 if p.best_chrome.shield_turn == 0 else 0

    col_p_info = GetCollisionInfo(p)
    col_op_info = GetCollisionInfo(op)
    col_prey_info = GetCollisionInfo(prey_pod)
    col_defender_info = GetCollisionInfo(defender_pod)

    col_op = WillPodsCollide(*col_p_info,*col_op_info)
    col_prey = WillPodsCollide(*col_p_info,*col_prey_info)
    col_defender = WillPodsCollide(*col_p_info,*col_defender_info)

    if col_op:
        print("collide with other:",round(abs(col_p_info[2]-col_op_info[2])+abs(col_p_info[3]-col_op_info[3])),file=db)
    elif col_prey:
        print('collide with prey:',round(abs(col_p_info[2]-col_prey_info[2])+abs(col_p_info[3]-col_prey_info[3])),file=db)   
    elif col_defender:
        print('collide with defender:',round(abs(col_p_info[2]-col_defender_info[2])+abs(col_p_info[3]-col_defender_info[3])),file=db)   

    if game_counter == 0:
        p.hdg = GetBoardAngle(p.point,cps[1].point)
        shield = 0
        turn = 0
    
    new_hdg = BoundAngle(p.hdg + turn)

    p.tx = p.x + 2000*m.cos(new_hdg)
    p.ty = p.y + 2000*m.sin(new_hdg)

    if shield == 1 and game_counter - p.last_shield_time <= 6:
        p.shield = 0
        p.tt = thrust
        p.message = 'hmmmm'
    elif shield == 1 and game_counter > 5:
        p.shield = 1
        p.tt = "SHIELD"
        p.message = 'SHIELD'
    elif game_counter - p.last_shield_time <= 3:
        p.shield = 0
        p.tt = thrust
        p.message = "wait"
    else:
        p.shield = 0
        p.tt = thrust
        p.message = ''
    
    
    if game_counter == 0:
        if p.num == 1 and p.ncd > 5000: p.tt = 'BOOST'
        if p.num == 2: p.tt = max_t*2/3
    elif game_counter < 3:
        if p.num == 1: p.tt = max_t
        if p.num == 2: p.tt = max_t*2/3
    elif p.last_checkpoint() and abs(p.nca_b) < rad(30):
        p.tt = max_t
        p.tx, p.ty = np.add(cps[p.nc].point,CalcTargetErrorDist(p))
        if abs(p.nca_b) < rad(8): p.tt = 'BOOST'

def GetCollisionInfo(pod):
    if pod.num == 1 or pod.num == 2:
        pod.StartSimulationMode()
        x1, y1 = pod.x, pod.y
        turn, thrust = ConvertGenToActual(pod.best_chrome.chrome[0])
        pod = MovePod(pod, turn, thrust)
        vx2, vy2 = pod.vx, pod.vy
        pod.ResetPod()
    else:
        x1, y1 = pod.x, pod.y
        nps = pod.next_pod_states[1]
        vx2, vy2 = nps['vx'], nps['vy']
    return (x1, y1, vx2, vy2)

def EvolvBestChrome(p, time_to_evolv, use_best_last=True, my_other_pod=None, add_chrome_to_pop=None):
    start_time = time.time()
    random_physics_chromes = [Chromesome(GetRandomChrome(future_moves_to_sim)) for _ in range(start_pop_random_chromes)]
    additional_chrome = add_chrome_to_pop if add_chrome_to_pop else None
    pop = [additional_chrome] + random_physics_chromes

    generation = 0
    max_scores = []
    perc_change_max_scores = []
    local_mutation_rate = 1
    while True:
        if not simulation and (time.time() - start_time) > time_to_evolv: break
        if simulation and generation > iteration_limit_for_sim: break

        if generation > 5: 
            local_mutation_rate = max(min(perc_change_max_scores[-1]/max(sum(perc_change_max_scores[:3])/3,0.0001),1),0.5)

        parents, children = [], []
        pop = GetPopulationRanking(p, pop, my_other_pod)
        
        pop.sort(key=lambda x: x.weighted_score, reverse=True)

        first_place_physics_chrome = pop[0].chrome
        second_place_physics_chrome = GetSecondPlaceChrome(pop)
        parents = [first_place_physics_chrome, second_place_physics_chrome]
        
        pop = RemoveLoserChromesIfApplicable(pop, max_chromes_to_keep)

        physics_child1, physics_child2 = CrossoverDirect(first_place_physics_chrome, second_place_physics_chrome)
        physics_child3 = CrossoverAverage(first_place_physics_chrome, second_place_physics_chrome)
        children = [physics_child1, physics_child2, physics_child3]

        mutated_physics_parents = [Mutate(x, global_mutation_rate*local_mutation_rate, odds_to_mutate_gain=global_odds_to_mutate/local_mutation_rate) for x in parents]
        mutated_physics_children = [Mutate(x, global_mutation_rate*local_mutation_rate, odds_to_mutate_gain=global_odds_to_mutate/local_mutation_rate) for x in children]

        if use_best_last and generation == 0 and len(p.best_chrome.chrome) > 1:
            best_last_turn_physics = Mutate(p.best_chrome.chrome[1:] + [GetRandomRawGene()], global_mutation_rate)
            pop += [Chromesome(best_last_turn_physics)]

        pop += [Chromesome(x) for x in mutated_physics_parents]
        pop += [Chromesome(x) for x in mutated_physics_children]

        pop = RemoveDuplicateChromes(pop)
        max_scores.append(max([round(x.weighted_score,3) for x in pop]))
        if generation > 1: perc_change_max_scores.append((max_scores[-1]-max_scores[-2])/max(max_scores[1],.00000001))
        generation += 1
    if game_counter > 0 and p == attacker_pod: 
        bio_div = max_scores.index(max(max_scores))*100/len(max_scores)
        # print('bio-diversity:',round(bio_div),'%', file=db)
        bio_div_log.append(bio_div)
        print('avg bio_diversity:',round(sum(bio_div_log)/len(bio_div_log)),'%',file=db)
    best_chrome = pop[0]
    print('ga_evoluations:',generation, file=db)
    # print('evolv_time:',round(time.time() - start_time,3),file=db)

    return best_chrome

def GetSecondPlaceChrome(chrome_pop):
    first_place = chrome_pop[0].chrome
    num_of_chromes = len(chrome_pop)

    for i in range(num_of_chromes):
        if i == num_of_chromes - 1: 
            #All chromes ended up being the same, so we're going to call second place a crossover with a random one
            second_place = CrossoverAverage(first_place, GetRandomChrome(future_moves_to_sim))
            break
        if chrome_pop[i+1].chrome != first_place:
            second_place = chrome_pop[i+1].chrome
            break

    return second_place

def RemoveDuplicateChromes(pop):
    # if len(pop) != len(set(pop)):
    #     raise Exception
    return list(set(pop))

def CrossoverDirect(fp_chrome, sp_chrome):
    child1_chrome, child2_chrome = copy.copy(fp_chrome), copy.copy(sp_chrome)
    turns = len(fp_chrome)

    crossover_point = random.randrange(1,turns)
    for i in range(crossover_point, turns):
        child1_chrome[i] = sp_chrome[i]
        child2_chrome[i] = fp_chrome[i]

    return (child1_chrome, child2_chrome)

def CrossoverAverage(fp_chrome, sp_chrome):
    fp, sp = fp_chrome, sp_chrome
    avg_chrome = []
    for x, y in zip(fp,sp):
        avg_move = []
        for h, g in zip(x,y):
            avg = (h+g)/2
            avg_move.append(avg)
        avg_chrome.append(tuple(avg_move))

    return avg_chrome

def Mutate(chrome, percent_mutation=1, odds_to_mutate_gain=1):
    mutated_chrome = []
    turns = len(chrome)
    which_ones_to_modify = [1 if random.random() < (1/turns) * odds_to_mutate_gain else 0 for i in range(turns)]
    if not any(which_ones_to_modify): 
        which_ones_to_modify[random.randrange(1,turns)] = 1
    
    for j, (turn, thrust) in enumerate(chrome):
        if which_ones_to_modify[j] == 1:
            random_turn, random_thrust = GetRandomRawGene()

            mutated_turn = turn + (random_turn - turn) * percent_mutation
            mutated_thrust = thrust + (random_thrust - thrust) * percent_mutation
            mutated_gene = (mutated_turn, mutated_thrust)

            mutated_chrome.append(mutated_gene)
        else:
            mutated_chrome.append((turn, thrust))
    return mutated_chrome

def RemoveLoserChromesIfApplicable(chrome_pop, max_chromes_to_keep):
    del chrome_pop[max_chromes_to_keep:]
    return chrome_pop

def GetPopulationRanking(p, pop, op = None):
    for Chrome in pop:
        if not Chrome.scores:        #This chrome hasn't been simulated yet.
            chrome_score = GetChromeScore(p, Chrome, op, pop)
            weighted_chrome_score = GetWeightedChromeScore(chrome_score)
            Chrome.scores = chrome_score
            Chrome.weighted_score = weighted_chrome_score
    return pop

def GetChromeScore(p, Chrome, op, pop):
    chrome_score = []
    prey_pod_states = copy.copy(prey_pod.next_pod_states)
    defender_pod_states = copy.copy(defender_pod.next_pod_states)

    p.StartSimulationMode()
    if op: 
        op.StartSimulationMode()
        op_Chrome = op.best_chrome
        op_chrome = op_Chrome.chrome
    else:
        op_chrome = [(0,0) for i in range(future_moves_to_sim)]

    collision_happened = False
    for turn, (gene_raw_p,gene_raw_op) in enumerate(zip(Chrome.chrome,op_chrome)):
        gene_act_p = ConvertGenToActual(gene_raw_p)
        hdg_change, thrust = gene_act_p
        if op: 
            gene_act_op = ConvertGenToActual(gene_raw_op)
            hdg_change_op, thrust_op = gene_act_op

        #Get pre-movement info
        x1, y1 = p.x, p.y
        if op: xop1, yop1 = op.x, op.y
        prey_x1, prey_y1 = prey_pod_states[turn]['x'], prey_pod_states[turn]['y']
        defender_x1, defender_y1 = defender_pod_states[turn]['x'], defender_pod_states[turn]['y']

        #Check if shielded
        sh2 = 1 if Chrome.shield_turn == turn else 0 
        if op: shop2 = 1 if op_Chrome.shield_turn == turn else 0

        #Move my pod
        p = MovePod(p, hdg_change, thrust, sh2)
        if op: op = MovePod(op, hdg_change_op, thrust_op, shop2)

        #Get post movement info
        vx2, vy2 = p.vx/0.85, p.vy/0.85
        if op: vxop2, vyop2 = op.vx/0.85, op.vy/0.85
        prey_vx2, prey_vy2, prey_sh2 = prey_pod_states[turn+1]['vx']/0.85, prey_pod_states[turn+1]['vy']/0.85, prey_pod_states[turn+1]['shield']
        defender_vx2, defender_vy2, defender_sh2 = defender_pod_states[turn+1]['vx']/0.85, defender_pod_states[turn+1]['vy']/0.85, defender_pod_states[turn+1]['shield']

        prey_sh2 = EnemyShieldOrNot(prey_vx2, prey_vy2, vx2, vy2)
        defender_sh2 = EnemyShieldOrNot(defender_vx2, defender_vy2, vx2, vy2)
        
        if not collision_happened and (turn < future_moves_to_sim):
            CollideWithOp = WillPodsCollide(x1, y1, vx2, vy2, xop1, yop1, vxop2, vyop2) if op else False
            if not CollideWithOp:
                CollideWithDefender, CollideWithPrey = False, False
                CollideWithDefender = WillPodsCollide(x1, y1, vx2, vy2, defender_x1, defender_y1, defender_vx2, defender_vy2)
                if not CollideWithDefender:
                    CollideWithPrey = WillPodsCollide(x1, y1, vx2, vy2, prey_x1, prey_y1, prey_vx2, prey_vy2)
            
            if CollideWithOp:
                collision_happened = True
                p1, v1, p2, v2 = MovePodsWithCollision(x1, y1, vx2, vy2, sh2, xop1, yop1, vxop2, vyop2, shop2)
                p.x, p.y, p.vx, p.vy = p1[0], p1[1], v1[0], v1[1]
                op.x, op.y, op.vx, op.vy = p2[0], p2[1], v2[0], v2[1]
            elif CollideWithDefender:
                collision_happened = True
                p1, v1, p2, v2 = MovePodsWithCollision(x1, y1, vx2, vy2, sh2, \
                                                defender_x1, defender_y1, defender_vx2, defender_vy2, defender_sh2)
                p.x, p.y, p.vx, p.vy = p1[0], p1[1], v1[0], v1[1]
                defender_pod_states = UpdateEnemyFuturePodStates(defender_pod, defender_pod_states, turn+1, (p2[0], p2[1], v2[0], v2[1]))
            elif CollideWithPrey:
                collision_happened = True
                p1, v1, p2, v2 = MovePodsWithCollision(x1, y1, vx2, vy2, sh2, \
                                            prey_x1, prey_y1, prey_vx2, prey_vy2, prey_sh2)
                p.x, p.y, p.vx, p.vy = p1[0], p1[1], v1[0], v1[1]
                prey_pod_states = UpdateEnemyFuturePodStates(prey_pod, prey_pod_states, turn+1, (p2[0], p2[1], v2[0], v2[1]))
            
            if sh2 == 0 and collision_happened and len(pop) < 10 and p.game_counter - p.last_shield_time > 6:
                pop.append(Chromesome(Chrome.chrome,shield_turn=turn))
        
        chrome_score.append(GetPodStateScore(p, op, prey_pod_states[turn+1]))

    p.ResetPod()
    if op: op.ResetPod()
    return chrome_score

def EnemyShieldOrNot(vx1, vy1, vx2, vy2):
    dvx = abs(vx1 - vx2)
    dvy = abs(vy1 - vy2)
    sum = dvx + dvy
    if sum > max_combined_vel_for_enemy_shield:
        return 1
    else:
        return 0

def GetPodStateScore(p, op, prey_pod_state):
    if op:
        prey_rank = prey_pod_state['rank']
        if p == racer_pod:
            #This function was called with 'p' being the racer. Just evolv a better racer.
            racer = p
            attacker = op
        else:
            #This function was called 'p' being the attacker. Evolv a better racer mostly, but then try to stop the prey as well.
            racer = op
            attacker = p

        racer_rank = racer.rank
        attacker_rank = attacker.rank
        if p == racer_pod:
            if prey_rank > racer_rank:
                return racer_rank - prey_rank*10
            else:
                return racer_rank
        else:
            return attacker_rank + racer_rank*10 - prey_rank*50
    else:
        return p.rank

def GetWeightedChromeScore(chrome_score):
    weighted_scores = [x * i for i, x in enumerate(chrome_score)]
    total_score = sum(weighted_scores)
    return total_score

def UpdateEnemyFuturePodStates(pod, pod_states, turn_to_update, info):

    existing_states = copy.copy(pod_states[:turn_to_update])
    current_state = copy.copy(pod_states[turn_to_update])
    future_states = []

    current_state['x'], current_state['y'] = info[0], info[1]
    current_state['vx'], current_state['vy'] = info[2], info[3]

    total_moves = len(pod_states) - 1
    future_moves_needed = total_moves - turn_to_update
    if future_moves_needed != 0:
        future_states = GetEnemyPodStates(pod, total_moves, starting_game_turn=turn_to_update)

    new_enemy_pod_states = existing_states + [current_state] + future_states

    return new_enemy_pod_states

def MovePodsWithCollision(p1_x1, p1_y1, p1_vx2, p1_vy2, p1_sh, p2_x1, p2_y1, p2_vx2, p2_vy2, p2_sh):
    #Find pre-impact point
    pod_sum_radius_sq = (pod_radius*2)**2

    end_vector_x = (p1_x1 + p1_vx2*0.85) - p2_vx2*0.85
    end_vector_y = (p1_y1 + p1_vy2*0.85) - p2_vy2*0.85
    
    start_coord = [p1_x1, p1_y1]
    end_coord = [end_vector_x, end_vector_y]

    closest_point = ClosestPointOnLine(start_coord, end_coord, [p2_x1, p2_y1])
    dist_closest_point_to_pod2_sq = DistSquared(closest_point, [p2_x1, p2_y1])
    back_dist = m.sqrt(pod_sum_radius_sq - dist_closest_point_to_pod2_sq)
    dist_closest_point_to_pod1 = GetDistance(start_coord, closest_point)

    total_vector_length = GetDistance(start_coord, end_coord)
    time = (dist_closest_point_to_pod1 - back_dist) / total_vector_length

    #move both pods to this point
    p1_x_pre_col = p1_x1 + p1_vx2 * time
    p1_y_pre_col = p1_y1 + p1_vy2 * time
    p2_x_pre_col = p2_x1 + p2_vx2 * time
    p2_y_pre_col = p2_y1 + p2_vy2 * time

    p1_mass = max(p1_sh * 10, 1)
    p2_mass = max(p2_sh * 10, 1)

    dvx = p1_vx2 - p2_vx2
    dvy = p1_vy2 - p2_vy2

    pre_col_dx = p1_x_pre_col - p2_x_pre_col
    pre_col_dy = p1_y_pre_col - p2_y_pre_col
    prod = pre_col_dx * dvx + pre_col_dy * dvy
    mass_coeff = (p1_mass + p2_mass) / (p1_mass * p2_mass)
    fx = (pre_col_dx * prod) / (pod_sum_radius_sq * mass_coeff)
    fy = (pre_col_dy * prod) / (pod_sum_radius_sq * mass_coeff)

    p1_vx_post_col = p1_vx2 - fx / p1_mass
    p1_vy_post_col = p1_vy2 - fy / p1_mass

    p2_vx_post_col = p2_vx2 + fx / p2_mass
    p2_vy_post_col = p2_vy2 + fy / p2_mass

    impulse = m.sqrt(fx**2 + fy**2)
    if impulse < 120:
        fx *= 120 / impulse
        fy *= 120 / impulse

    p1_vx_post_col -= fx / p1_mass
    p1_vy_post_col -= fy / p1_mass
    p2_vx_post_col += fx / p2_mass
    p2_vy_post_col += fy / p2_mass

    p1_x_post_col = p1_x_pre_col + p1_vx_post_col * (1 - time)
    p1_y_post_col = p1_y_pre_col + p1_vy_post_col * (1 - time)
    p2_x_post_col = p2_x_pre_col + p2_vx_post_col * (1 - time)
    p2_y_post_col = p2_y_pre_col + p2_vy_post_col * (1 - time)

    p1x, p1y = p1_x_post_col, p1_y_post_col
    p1vx, p1vy = p1_vx_post_col * 0.85, p1_vy_post_col * 0.85
    p2x, p2y = p2_x_post_col, p2_y_post_col
    p2vx, p2vy = p2_vx_post_col * 0.85, p2_vy_post_col * 0.85

    return ([p1x,p1y],[p1vx,p1vy],[p2x,p2y],[p2vx,p2vy])

def WillPodsCollide(ax1, ay1, avx2, avy2, bx1, by1, bvx2, bvy2, note=''):
    dist_sq_start_points = DistSquared([ax1, ay1], [bx1, by1])
    if dist_sq_start_points < (pod_radius*2)**2 or dist_sq_start_points > ((pod_radius*2)*3)**2: 
        # If both pods are already over-lapping before they even move.
        # This happens because there multiple collisions and it's not
        # being perfectly modeled.
        # Also checking to see if pods are so far apart that we don't even need to check. somewhat arbitrary 800*3**2
        return False
    
    start_coord = [ax1, ay1]
    end_coord = [(ax1 + avx2*0.85) - bvx2*0.85, (ay1 + avy2*0.85) - bvy2*0.85]
    collision_happened = CheckPodHitCircle(start_coord, end_coord, pod_radius, [bx1, by1], pod_radius)

    return collision_happened

def DistToTargetV(vnc_vel,max_v_at_cp):
    if vnc_vel > 0: #check for division by zero
        return vnc_vel/m.log(0.85)*(0.85**(m.log10(max_v_at_cp/vnc_vel)/m.log10(0.85))-1)
    else:
        return 0

def TurnAtCheckpointLogic(p):
    va = GetBoardAngle(p.point, [p.x+p.vx, p.y+p.vy])
    nnca_vel = BoundAngle(p.nnca - va)

    max_v_at_cp = 2000*8**(-(abs(nnca_vel))) + 100
    if abs(nnca_vel) < rad(10): max_v_at_cp = 2000
    
    v = m.sqrt(p.vx**2 + p.vy**2)
    nca_vel = ToBody(p.nca - va)
    vnc_vel = v * m.cos(abs(nca_vel))
    
    ncd_v_at_cp = DistToTargetV(vnc_vel,max_v_at_cp)
      
    if ncd_v_at_cp > (p.ncd) and HitCheckpoint(p,cps[p.nc].point):
        return True
    else:
        return False

def HitCheckpoint(p,target_point):
    cx, cy = target_point
    x1, y1 = p.x - cx, p.y - cy
    x2, y2 = p.x + p.vx - cx, p.y + p.vy - cy
    dx = x2 - x1
    dy = y2 - y1
    dr_sq = dx**2 + dy**2
    D = x1 * y2 - x2 * y1
    delta = checkpoint_radius**2 * dr_sq - D**2
    hcp = True if delta > 0 else False
    return hcp

def CalcTargetErrorDist(p):
    va = GetBoardAngle(p.point, [p.x+p.vx, p.y+p.vy])
    nca_vel = ToBody(p.nca - va)

    bearing_from = BoundAngle(p.nca + m.pi)

    if nca_vel < 0:
        perp_angle = BoundAngle(bearing_from + m.pi/2)
    else:
        perp_angle = BoundAngle(bearing_from - m.pi/2)

    max_dist = max(min(p.ncd/10000,1)*3500,500)

    target_error_dist = min(max(int(m.tan(abs(nca_vel))*p.ncd),0),max_dist)

    x_error = int(m.cos(perp_angle)*target_error_dist)
    y_error = int(m.sin(perp_angle)*target_error_dist)

    return (x_error, y_error)

def GetPerfectPhysicsChrome(p, moves):
    physics_chrome_raw = []
    p.StartSimulationMode()
    for _ in range(moves):
        tx_cp, ty_cp = cps[p.nc].x, cps[p.nc].y
        target_error_x, target_error_y = CalcTargetErrorDist(p)

        tx = tx_cp + target_error_x
        ty = ty_cp + target_error_y
        tt = GetDirectModeThrust(p.point, cps[p.nc].point, p.hdg)

        TACP = TurnAtCheckpointLogic(p)
        if TACP:
            tx, ty = cps[p.nnc].point
            tt = 0

        if p.ncd < 1200 and not HitCheckpoint(p,cps[p.nc].point): tt = 0

        hdg_change = GetHdgChange(p.point, [tx, ty], p.hdg)
        
        gene_raw = ConvertGenToRaw(hdg_change, tt)
        physics_chrome_raw.append(gene_raw)
        
        p = MovePod(p, hdg_change, tt)
    p.ResetPod()
    return physics_chrome_raw

def ConvertGenToRaw(hdg_change, thrust):
    hdg_chg_raw = (hdg_change/rad(18)+1) / 2
    thrust_raw = thrust/max_t
    return (hdg_chg_raw, thrust_raw)

def GetRandomChromes(num_of_chromes, moves):
    chromes_raw = []
    for _ in range(num_of_chromes):
        chromes_raw.append(GetRandomChrome(moves))
    return chromes_raw

def GetRandomChrome(moves):
    chrome_raw = []
    for _ in range(moves):
        chrome_raw.append(GetRandomRawGene())
    return chrome_raw

def GetRandomRawGene():
    turn = random.choice([random.random()*turn_gain, random.random()/turn_gain])
    thrust = random.choice([random.random()*thrust_gain, random.random()/thrust_gain])
    return (turn, thrust)

def ConvertGenToActual(gen):
    turn, thrust = gen

    rad_18 = rad(18)

    turn = max(min((turn*2 - 1) * rad_18, rad_18), -rad_18)
    thrust = max(min(max_t * thrust, max_t), 0)
    
    return (turn, thrust)

def GetOpponentNextMoves(moves=1):
    global direct_bot_states
    next_pod_states = {}
    direct_bot_states[1] = GetDirectBotNextPositions(pods[1], moves)
    direct_bot_states[2] = GetDirectBotNextPositions(pods[2], moves)
    next_pod_states[3] = GetEnemyPodStates(pods[3], moves)
    next_pod_states[4] = GetEnemyPodStates(pods[4], moves)
    return next_pod_states

def GetEnemyPodStates(p, moves, starting_game_turn=0):
    pod_states = []

    #Current state
    if starting_game_turn == 0:
        pod_state_stats = p.SavePodStateStats()
        pod_states.append(pod_state_stats)

    p.StartSimulationMode()
    for i in range(starting_game_turn, moves):
        prev_my_pod1_state = direct_bot_states[1][i]
        prev_my_pod2_state = direct_bot_states[2][i]
        inputs = GetNNinput(p, prev_my_pod1_state, prev_my_pod2_state)
        inputs_norm = NormalizeNNinputs(inputs)
        output = NN.forward(inputs_norm, 1)
        hdg_change, thrust_used = output[0][0], output[0][1]
        hdg_change = hdg_change * rad(36) - rad(18)
        thrust_used *= max_t
        pod_state = MovePod(p,hdg_change,thrust_used)
        pod_state_stats = pod_state.SavePodStateStats()
        pod_states.append(pod_state_stats)
    p.ResetPod()
    return pod_states

def NormalizeNNinputs(inputs):
    input_min = input_normalization[0]
    input_max = input_normalization[1]
    input_norm = np.clip(np.round((inputs - input_min) / (input_max - input_min),2),0,1)
    return input_norm

def MovePod(p, hdg_change, thrust, shield=None):
    p.hdg = BoundAngle(p.hdg + hdg_change)

    if shield==1: 
        thrust = 0
        p.shield = 1
    elif (p.game_counter - p.last_shield_time) <= 3:
        p.shield = 0
        thrust = 0
    else:
        p.shield = 0

    vx = m.cos(p.hdg)*thrust + p.vx #vx used to calc new x
    vy = m.sin(p.hdg)*thrust + p.vy #vy used to calc new y
    p.x = p.x + vx
    p.y = p.y + vy

    if DistSquared(p.point,cps[p.nc].point) < checkpoint_radius**2:
        if p.nc == checkpoint_count - 1:
            p.nc = 0
        else:
            p.nc += 1

    p.vx = vx * 0.85
    p.vy = vy * 0.85
    
    p.game_counter += 1

    return p

def GetNNinput(p, my_pod1_state, my_pod2_state):
    export_list_input = []
    q = 3 if p.num == 4 else 4
    export_list_input.append(0 if p.rank < pods[q].rank else 1)
    export_list_input.append(p.vx)
    export_list_input.append(p.vy)
    export_list_input.append(round(p.nca_b,2))
    export_list_input.append(int(p.ncd))
    export_list_input.append(round(p.nnca_b,2))
    export_list_input.append(int(p.nncd))
    export_list_input.append(0 if my_pod1_state['rank'] < my_pod2_state['rank'] else 1)
    export_list_input.append(round(np.clip(my_pod1_state['rank'], 0, 1),2))
    export_list_input.append(round(my_pod1_state['hdg'],2))
    export_list_input.append(round(ToBody(my_pod1_state['hdg'] - p.hdg),2))
    export_list_input.append(int(GetDistance([my_pod1_state['x'],my_pod1_state['y']], p.point)))
    export_list_input.append(int(p.vx - my_pod1_state['vx']))
    export_list_input.append(int(p.vy - my_pod1_state['vy']))
    export_list_input.append(round(np.clip(my_pod2_state['rank'], 0, 1),2))
    export_list_input.append(round(my_pod2_state['hdg'],2))
    export_list_input.append(round(ToBody(my_pod2_state['hdg'] - p.hdg),2))
    export_list_input.append(int(GetDistance([my_pod2_state['x'], my_pod2_state['y']], p.point)))
    export_list_input.append(int(p.vx - my_pod2_state['vx']))
    export_list_input.append(int(p.vy - my_pod2_state['vy']))
    return export_list_input

def GetDirectBotNextPositions(p, moves):
    pod_states = []

    pod_state_stats = p.SavePodStateStats()
    pod_states.append(pod_state_stats)

    p.StartSimulationMode()
    for _ in range(moves):
        pod_state = MovePodDirectMode(p, cps[p.nc].x, cps[p.nc].y)
        pod_state_stats = pod_state.SavePodStateStats()
        pod_states.append(pod_state_stats)
    p.ResetPod()
    return pod_states

def MovePodDirectMode(p, tx, ty):
    tt = GetDirectModeThrust(p.point, [tx,ty], p.hdg)
    hdg_change = GetHdgChange(p.point, [tx, ty], p.hdg)
    pod = MovePod(p, hdg_change, tt)
    return pod

def GetHdgChange(a,b,current_hdg):
    o = GetBoardAngle(a,b) - current_hdg
    n = max(ToBody(o),-18*m.pi/180)
    return min(n,18*m.pi/180)

def GetDirectModeThrust(p_point,target_point,hdg):
    t = max_t
    angle_to_target = abs(ToBody(BoundAngle(GetBoardAngle(p_point,target_point)-hdg)))
    if angle_to_target < rad(30):
        t = max_t
    elif angle_to_target < rad(45):
        t = max_t
    elif angle_to_target < rad(90):
        t = max_t*0.25
    elif angle_to_target < rad(135):
        t = max_t*0.125
    else:
        t = 0 
    return int(min(max(t,0),max_t))

def CheckPodHitCircle(pod_start_coord, pod_end_coord, pod_radius, \
                                circle_coord, circle_radius):
    #This checks if a moving pod will intersect with a circle
    #The circle could be another pod or a checkpoint
    #If the circle is a checkpoint, the center of the moving
    #pod will have to intersect, so use a pod_radius = 0
    #If the circle is another pod, use a pod radius = 400
    closest_point = ClosestPointOnLine(pod_start_coord, pod_end_coord, circle_coord)
    # cx is the x coord on the line segment closest to the circle_coord
    # cy is the y coord on the line segment closest to the circle_coord
    
    if DistSquared(circle_coord,closest_point) < (circle_radius + pod_radius)**2:
        if CheckPointInSegment(pod_start_coord, pod_end_coord, closest_point):
            intersection = True
        elif DistSquared(closest_point, pod_end_coord) < (pod_radius+circle_radius)**2:
            intersection = True
        else:
            intersection = False
    else:
        intersection = False
    return intersection

def ClosestPointOnLine(start_seg, end_seg, point):
    lx1, ly1 = start_seg
    lx2, ly2 = end_seg
    x0, y0 = point
    A1 = ly2 - ly1
    B1 = lx1 - lx2
    C1 = (ly2 - ly1)*lx1 + (lx1 - lx2)*ly1
    C2 = -B1*x0 + A1*y0
    det = A1**2 + B1**2
    cx, cy = 0, 0
    if det != 0:
        cx = ((A1*C1 - B1*C2)/det)
        cy = ((A1*C2 + B1*C1)/det)
    return (cx, cy)

def CheckPointInSegment(start_vector, end_vector, point):
    '''Check to see if a point lies on a line segment'''
    svx, svy = start_vector
    evx, evy = end_vector
    px, py = point

    if py - svy == 0 or evy - svy == 0: return False #Not sure how this would ever happen, but it does
    if not (((px - svx) / (py - svy)) - ((evx - svx) / (evy - svy)) < 2): return False
    if not (min(svx,evx) <= px <= max(svx,evx) and min(svy,evy) <= py <= max(svy,evy)): return False
    return True

def DistSquared(p1, p2):
    p1x, p1y = p1
    p2x, p2y = p2
    dist = (p2x - p1x)**2 + (p2y - p1y)**2
    return dist

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

def GameOutput():
    output = []
    for i in range(1,3):
        if pods[i].message == "":
            output.append(pods[i].tx + " " + pods[i].ty + " " + pods[i].tt)
        else:
            output.append(pods[i].tx + " " + pods[i].ty + " " + pods[i].tt + " " + pods[i].message)

    for i in range(len(output)):
        print(output[i])
    return (output[0], output[1])

def RaceDist():
    global total_race_dist
    global lap_dist
    cp_dist = 0
    l_dist = 0
    for i in range(len(cps)):
        cp1 = cps[i]
        if i+1 == checkpoint_count:
            cp2 = cps[0]
        else:
            cp2 = cps[i+1]
        cp_dist = GetDistance(cp1.point,cp2.point)
        cp1.dist_prior = l_dist
        cp2.dist_prior_cp = cp_dist
        l_dist = l_dist + cp_dist
    total_race_dist = l_dist * lap_count
    lap_dist = l_dist

def GetDistance(a,b):
    #Distance between two points, defined by x,y coordinates
    return m.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

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

def GetLeaderBoardAndRoles():
    global racer_pod, attacker_pod, prey_pod, defender_pod

    if pods[1].rank <= pods[2].rank:
        attacker_pod = pods[1]
        racer_pod = pods[2]
        pods[1].role = 'attacker'
        pods[2].role = 'racer'
    else:
        attacker_pod = pods[2]
        racer_pod = pods[1]
        pods[2].role = 'attacker'
        pods[1].role = 'racer'
    
    if pods[3].rank >= pods[4].rank:
        prey_pod = pods[3]
        defender_pod = pods[4]
        pods[3].role = 'prey'
        pods[4].role = 'defender'
    else:
        prey_pod = pods[4]
        defender_pod = pods[3]
        pods[4].role = 'prey'
        pods[3].role = 'defender'

class Neural_Network:
    def __init__(self, input_layer_size, output_layer_size, weights, bias_weights, \
                hidden_layer_sizes=[], activation_func = 'sigmoid', Lambda=0):        
        #Define Hyperparameters
        self.inputLayerSize = input_layer_size#len(inputs[0])
        self.outputLayerSize = output_layer_size#len(output[0])
        self.hiddenLayerSizes = hidden_layer_sizes
        self.layerSizes = [self.inputLayerSize] + self.hiddenLayerSizes + [self.outputLayerSize]
        self.num_layers = len(self.hiddenLayerSizes) + 2
        self.weights = weights
        self.bias_weights = bias_weights
        self.Lambda = Lambda

        if activation_func == 'ELU':
            self.activ_func = self.elu
            self.activ_func_prime = self.eluPrime
        elif activation_func == 'RELU':
            self.activ_func = self.relu
            self.activ_func_prime = self.reluPrime
        else:
            self.activ_func = self.sigmoid
            self.activ_func_prime = self.sigmoidPrime
        
    def forward(self, X, train_items=-1):
        #Propogate inputs though network
        self.z = []
        self.a = []
        if train_items == -1:
            train_sets = X.shape[0]
        else:
            train_sets = 1

        for i, w in enumerate(self.weights):
            bias_vector = np.ones((train_sets,1))
            bias = np.dot(bias_vector, self.bias_weights[i])
            if i == 0:
                self.z.append(np.dot(X, w) + bias)
                self.a.append(self.activ_func(self.z[-1]))
            else:
                self.z.append(np.dot(self.a[-1], w) + bias)
                self.a.append(self.activ_func(self.z[-1]))
        
        yHat = self.a[-1]
        return yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def elu(self, z):
        alpha = 1.5
        return np.where(z <= 0, alpha*(np.exp(z) - 1), z)
    
    def eluPrime(self, z):
        alpha = 1.5
        return np.where(z <= 0, self.elu(z) + alpha, 1)

    def relu(self, z):
        return np.where(z < 0, 0, z)

    def reluPrime(self, z):
        return np.where(z < 0, 0, 1)

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        train_sets = X.shape[0] if self.Lambda != 0 else 1
        J = (1/2)*np.sum((y-self.yHat)**2)/train_sets + (self.Lambda/2) * np.sum([np.sum(x**2) for x in self.weights])
        return J
        
    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        delta = []
        dJdW = []
        dJdW_bias = []
        train_sets = X.shape[0] if self.Lambda != 0 else 1
        
        for i in range(self.num_layers - 2,-1,-1):
            if i == self.num_layers - 2:
                delta.insert(0, np.multiply(-(y-self.yHat), self.activ_func_prime(self.z[i])))
            else:
                delta.insert(0, np.dot(delta[0], self.weights[i+1].T) * self.activ_func_prime(self.z[i]))

            if i == 0:
                dJdW.insert(0, np.dot(X.T, delta[0])/train_sets + self.Lambda * self.weights[i])
            else:
                dJdW.insert(0, np.dot(self.a[i-1].T, delta[0])/train_sets + self.Lambda * self.weights[i])
            
            bias_vector = np.ones((1,X.shape[0]))
            dJdW_bias.insert(0, np.dot(bias_vector, delta[0])/train_sets + self.Lambda * self.bias_weights[i]) #Bias dJdW updated as just the delta

        return (dJdW, dJdW_bias)
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        temp = []
        for x, y in zip(self.weights, self.bias_weights):
            temp.append(np.concatenate((x.ravel(), y.ravel())))
        params = np.concatenate(temp)
        return params
    
    def setParams(self, params):
        
        W_start, W_end = 0, 0
        W_start_bias, W_end_bias = 0, 0
        for i in range(self.num_layers - 1):
            front_end_layer, back_end_layer = i, i + 1

            W_end = W_start + self.layerSizes[front_end_layer] * self.layerSizes[back_end_layer]
            W_start_bias = W_end
            W_end_bias = W_start_bias + self.layerSizes[back_end_layer]
            try:
                self.weights[i] = np.reshape(params[W_start:W_end], \
                    (self.layerSizes[front_end_layer], self.layerSizes[back_end_layer]))
            except:
                pass
            self.bias_weights[i] = np.reshape(params[W_start_bias:W_end_bias], \
                (1, self.layerSizes[back_end_layer]))
            W_start = W_end_bias

    def computeGradients(self, X, y):
        dJdW, dJdW_bias = self.costFunctionPrime(X, y)
        temp = []
        for x, y in zip(dJdW, dJdW_bias):
            temp.append(np.concatenate((x.ravel(), y.ravel())))
        grad = np.concatenate(temp)
        return grad


class Checkpoint():
    def __init__(self, num_, x_, y_):
        self.num = num_
        self.x = x_
        self.y = y_
    
    @property
    def point(self):
        return [self.x, self.y]

class Chromesome():
    def __init__(self, chrome, scores = [], weighted_score = -m.inf, shield_turn = -1):
        self.chrome = chrome
        self.scores = scores
        self.weighted_score = weighted_score
        self.shield_turn = shield_turn

    def __eq__(self, other):
        return [self.chrome,self.shield_turn] == [other.chrome, other.shield_turn]

    def __hash__(self):
        return hash(self.weighted_score)

class Pod():
    def __init__(self, num):
        self.num = num
        self._x, self._y = 0, 0
        self._vx, self._vy = 0, 0
        self._hdg = 0
        self._init_hdg = 0
        self._nc = 0
        self.lap = 0
        self.prev_cp = 0
        self.nnc = 0
        self._tx = str(0) #target x
        self._ty = str(0) #target y
        self._tt = str(0) #target thrust
        self.message = ""
        self.next_pod_states = []
        self._shield = 0
        self.last_shield_time = -5
        self._game_counter = 0
        self.cp_timer = 100
        self.last_used_chrome = []
        self.sim_mode_on = False
        self.sim_game_counter = []
        self.sim_x = []
        self.sim_y = []
        self.sim_vx = []
        self.sim_vy = []
        self.sim_nc = []
        self.sim_prev_cp = []
        self.sim_cp_timer = []
        self.sim_lap = []
        self.sim_nnc = []
        self.sim_hdg = []
        self.sim_shield = []
        self.sim_last_shield_time = []
        self.sim_next_pod_states = []
        self.best_chrome = Chromesome([(0,0,0)])
        self.rank_dict = {}

    @property
    def point(self):
        return [self._x,self._y]

    @property
    def shield(self):
        return self._shield
    @shield.setter
    def shield(self,value):
        if value == 1: self.last_shield_time = self.game_counter
        self._shield = value

    @property
    def game_counter(self):
        return self._game_counter
    @game_counter.setter
    def game_counter(self,value):
        self.cp_timer -= 1
        self._game_counter = value

    @property
    def nc(self):
        return self._nc
    @nc.setter
    def nc(self, value):
        if self._nc != value:
            self.prev_cp = self._nc
            self.cp_timer = 100
            if value == 1:
                self.lap += 1
        
        if value == checkpoint_count-1:
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
        self._tx = str(int(round(value)))

    @property
    def ty(self):
        return self._ty
    @ty.setter
    def ty(self,value):
        self._ty = str(int(round(value)))

    @property
    def tt(self):
        return self._tt
    @tt.setter
    def tt(self,value):
        if str(value).isalpha():
            self._tt = value
        else:
            self._tt = str(int(round(value)))

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
    def nca(self):
        return GetBoardAngle(self.point, cps[self.nc].point)

    @property
    def nca_b(self):
        return ToBody(self.nca - self._hdg)

    @property
    def nnca(self):
        return GetBoardAngle(self.point, cps[self.nnc].point)

    @property
    def nnca_b(self):
        return ToBody(self.nnca - self._hdg)

    @property
    def nncd(self):
        return GetDistance(self.point, cps[self.nnc].point)

    @property
    def init_hdg(self):
        return self._init_hdg
    @init_hdg.setter
    def init_hdg(self,value):
        self._init_hdg = value
        self._hdg = rad(value)

    @property
    def hdg(self):
        return self._hdg
    @hdg.setter
    def hdg(self,value):
        self._hdg = value

    @property
    def rank(self):
        unique_key = str(self.game_counter)+str(self.x)+str(self.y)
        try:
            return self.rank_dict[unique_key]
        except:
            pass

        l_dist = (self.lap-1) * lap_dist
        cp_passed_dist = cps[self.prev_cp].dist_prior
        nc = GetDistance(cps[self.nc].point,cps[self.prev_cp].point) - self.ncd

        dist_gone = l_dist + cp_passed_dist + nc
        percent_complete = dist_gone / total_race_dist

        self.rank_dict[unique_key] = percent_complete
        return percent_complete

    @property
    def ncd(self):
        return GetDistance(self.point, cps[self.nc].point)

    def last_checkpoint(self):
        if self.lap == lap_count and self.nc == 0:
            return True
        else:
            return False

    def StartSimulationMode(self):
        self.sim_mode_on = True
        self.sim_game_counter.append(self.game_counter)
        self.sim_x.append(self._x)
        self.sim_y.append(self._y)
        self.sim_vx.append(self._vx)
        self.sim_vy.append(self._vy)
        self.sim_nc.append(self._nc)
        self.sim_prev_cp.append(self.prev_cp)
        self.sim_cp_timer.append(self.cp_timer)
        self.sim_lap.append(self.lap)
        self.sim_nnc.append(self.nnc)
        self.sim_hdg.append(self._hdg)
        self.sim_shield.append(self._shield)
        self.sim_last_shield_time.append(self.last_shield_time)
    
    def ResetPod(self):
        if self.sim_mode_on:
            self._game_counter = self.sim_game_counter[-1]
            del self.sim_game_counter[-1]
            self._x = self.sim_x[-1]
            del self.sim_x[-1]
            self._y = self.sim_y[-1]
            del self.sim_y[-1]
            self._vx = self.sim_vx[-1]
            del self.sim_vx[-1]
            self._vy = self.sim_vy[-1]
            del self.sim_vy[-1]
            self._nc = self.sim_nc[-1]
            del self.sim_nc[-1]
            self.prev_cp = self.sim_prev_cp[-1]
            del self.sim_prev_cp[-1]
            self.cp_timer = self.sim_cp_timer[-1]
            del self.sim_cp_timer[-1]
            self.lap = self.sim_lap[-1]
            del self.sim_lap[-1]
            self.nnc = self.sim_nnc[-1]
            del self.sim_nnc[-1]
            self._hdg = self.sim_hdg[-1]
            del self.sim_hdg[-1]
            self._shield = self.sim_shield[-1]
            del self.sim_shield[-1]
            self.last_shield_time = self.sim_last_shield_time[-1]
            del self.sim_last_shield_time[-1]

            if not self.sim_game_counter: self.sim_mode_on = False

    def SavePodStateStats(self):
        stats = {}
        stats['game_counter'] = self.game_counter
        stats['x'] = self._x
        stats['y'] = self._y
        stats['vx'] = self._vx
        stats['vy'] = self._vy
        stats['nc'] = self._nc
        stats['prev_cp'] = self.prev_cp
        stats['cp_timer'] = self.cp_timer
        stats['lap'] = self.lap
        stats['nnc'] = self.nnc
        stats['hdg'] = self._hdg
        stats['shield'] = self._shield
        stats['last_shield_time'] = self.last_shield_time
        stats['rank'] = self.rank
        return stats

    def ReincarnatePodState(self, stats):
        self.StartSimulationMode()
        self.game_counter = stats['game_counter']
        self._x = stats['x']
        self._y = stats['y']
        self._vx = stats['vx']
        self._vy = stats['vy']
        self._nc = stats['nc']
        self.prev_cp = stats['prev_cp']
        self.cp_timer = stats['cp_timer']
        self.lap = stats['lap']
        self.nnc = stats['nnc']
        self._hdg = stats['hdg']
        self._shield = stats['shield']
        self.last_shield_time = stats['last_shield_time']
        return self

if not simulation:
    lap_count = int(input())
    checkpoint_count = int(input())
    for j in range(checkpoint_count):
        x_, y_ = [int(i) for i in input().split()]
        cps[j] = Checkpoint(j, x_, y_)
else:
    lap_count = simulator.GetLapCount()
    checkpoint_count = simulator.GetCheckpointCount()
    for j in range(checkpoint_count):
        x_, y_ = simulator.GetCheckpointInfo(j)
        cps[j] = Checkpoint(j, x_, y_)

RaceDist()
for i in range(1, num_of_pods + 1):
    pods[i] = Pod(i)

NN = Neural_Network(input_layer_size, output_layer_size, weights, bias_weights, \
        hidden_layer_sizes, activation_func, Lambda)

game_counter = 0
while StillPlaying:
    start_total_time = time.time()

    if not simulation:
        for i in range(1,len(pods)+1):
            pods[i].game_counter = game_counter
            pods[i].x, pods[i].y, pods[i].vx, pods[i].vy, \
                pods[i].init_hdg, pods[i].nc = [int(i) for i in input().split()]
    else:
        for i in range(1,len(pods)+1):
            pods[i].game_counter = game_counter
            pods[i].x, pods[i].y, pods[i].vx, pods[i].vy, \
                pods[i].hdg, pods[i].nc = simulator.GetPodInfo(pods[i].num)
            if game_counter == 0: pods[i].nc = 1
    
    GetLeaderBoardAndRoles()

    opp_next_moves = GetOpponentNextMoves(moves=future_moves_to_sim)
    pods[3].next_pod_states = opp_next_moves[3]
    pods[4].next_pod_states = opp_next_moves[4]

    #Evolv my racer
    perfect_physics_chrome_racer = Chromesome(GetPerfectPhysicsChrome(racer_pod, future_moves_to_sim))
    racer_pod.best_chrome = EvolvBestChrome(racer_pod, physics_evolv_time, use_best_last=True, \
                            my_other_pod=None, add_chrome_to_pop=perfect_physics_chrome_racer)

    #Evolv my attacker
    perfect_physics_chrome_attacker = Chromesome(GetPerfectPhysicsChrome(attacker_pod, future_moves_to_sim))
    attacker_pod.best_chrome = EvolvBestChrome(attacker_pod, physics_evolv_time, use_best_last=True, \
                            my_other_pod=racer_pod, add_chrome_to_pop=perfect_physics_chrome_attacker)

    #Evolv my racer again
    racer_pod.best_chrome = EvolvBestChrome(racer_pod, physics_evolv_time_again, use_best_last=False, \
                            my_other_pod=attacker_pod, add_chrome_to_pop=racer_pod.best_chrome)

    #Set Commands
    SetCommands(racer_pod, attacker_pod)
    SetCommands(attacker_pod, racer_pod)

    print('End time:',round(time.time() - start_total_time, 3), file = db)
    game_counter += 1
    game_output = GameOutput()
    if simulation:
        simulator.SendOutputs(game_output[0], game_output[1])
        StillPlaying = simulator.AreWeStillPlaying()
        # if game_counter > 0: StillPlaying = False
