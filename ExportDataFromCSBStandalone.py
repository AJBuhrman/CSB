#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#THIS GOES IN THE PARAMETER DECLARATIONS SECTION;
export_list_prey = []
export_list_def = []
last_game_turn_export_prey = []
last_game_turn_export_def = []
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def GetExportInfo(p):
    export_list = []
    hdg_change = -1
    thrust_used = -1
    #i is the pod to export, j is the other opponent pod
    i = p.num
    j = 4 if i == 3 else 3
    export_list.append(0 if p.rank < pods[j].rank else 1)
    export_list.append(p.vx)
    export_list.append(p.vy)
    export_list.append(round(p.nca_b,2))
    export_list.append(int(p.ncd))
    export_list.append(round(p.nnca_b,2))
    export_list.append(int(p.nncd))
    export_list.append(0 if pods[1].rank < pods[2].rank else 1)
    export_list.append(round(np.clip(pods[1].rank, 0, 1),2))
    export_list.append(round(pods[1].hdg,2))
    export_list.append(round(ToBody(pods[1].hdg - p.hdg),2))
    export_list.append(int(GetDistance(pods[1].point, p.point)))
    export_list.append(int(p.vx - pods[1].vx))
    export_list.append(int(p.vy - pods[1].vy))
    export_list.append(round(np.clip(pods[2].rank, 0, 1),2))
    export_list.append(round(pods[2].hdg,2))
    export_list.append(round(ToBody(pods[2].hdg - p.hdg),2))
    export_list.append(int(GetDistance(pods[2].point, p.point)))
    export_list.append(int(p.vx - pods[2].vx))
    export_list.append(int(p.vy - pods[2].vy))
    
    if game_counter == 0:
        p.prev_turn_collision = False#CheckForProbableCollision(p)
        return (export_list,-1,-1)

    if game_counter > 0:
        hdg_change = round(ToBody(p.hdg-p.prev_game_turn_state.hdg),2)
        thrust_used = int(GetThrust(p.prev_game_turn_state,p,p.prev_turn_collision))
    p.prev_turn_collision = CheckForProbableCollision(p)

    return (export_list, hdg_change, thrust_used)

def CheckForProbableCollision(p):
    for pod in pods.values():
        if p.num == pod.num: continue
        
        #Other pod info
        if pod.next_pod_states:
            pod_x, pod_y = pod.x, pod.y
            pod_vx, pod_vy = pod.next_pod_states[1]['vx']/0.85, pod.next_pod_states[1]['vy']/0.85
        else:
            pod_x, pod_y = pod.x, pod.y
            if pod.tt == 'BOOST' or pod.tt == 'SHIELD':
                continue
            new_hdg = BoundAngle(GetHdgChange(pod.point,[int(pod.tx),int(pod.ty)],pod.hdg)+pod.hdg)
            pod_vx, pod_vy = pod.vx + m.cos(new_hdg)*int(pod.tt), pod.vy + m.sin(new_hdg)*int(pod.tt)
        
        #My pod info
        p_x, p_y = p.x, p.y
        if p.tt == 'BOOST' or p.tt == 'SHIELD':
            return False
        new_hdg = BoundAngle(GetHdgChange(p.point,[int(p.tx),int(p.ty)],p.hdg)+p.hdg)
        p_vx, p_vy = p.vx + m.cos(new_hdg)*int(p.tt), p.vy + m.sin(new_hdg)*int(p.tt)

        #Will they collide?
        if WillPodsCollide(p_x, p_y, p_vx, p_vy, pod_x, pod_y, pod_vx, pod_vy):
            return True

        return False

def GetThrust(old_p, new_p, collision_happened):
    #collision1 = GetCollision(p,pods[1],steps_in_future
    if collision_happened: return -1
    
    thrust = abs(((new_p.x - old_p.x) - old_p.vx) / m.cos(new_p.hdg))

    if 0 <= thrust <= max_t:
        return thrust
    else:
        return -1

def AssignPrevGameTurnState():
    for p in pods.values():
        p.prev_game_turn_state = None
        p.prev_game_turn_state = copy.copy(p)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#This goes in the game loop!! After setting the commands.
    export_list_input_prey, hdg_change_prey, thrust_used_prey = GetExportInfo(prey_pod)
    export_list_input_def, hdg_change_def, thrust_used_def = GetExportInfo(defender_pod)

    if export_list_input_prey and thrust_used_prey != -1:
        export_list_prey.append(last_game_turn_export_prey + [hdg_change_prey] + [thrust_used_prey])
    last_game_turn_export_prey = export_list_input_prey

    if export_list_input_def and thrust_used_def != -1:
        export_list_def.append(last_game_turn_export_def + [hdg_change_def] + [thrust_used_def])
    last_game_turn_export_def = export_list_input_def
    
    AssignPrevGameTurnState()

    print(game_counter,file=db)
    if game_counter % 16 == 0 or any([p.rank > 0.98 for p in pods.values()]): 
        for y in export_list_prey:
            print(*y,sep=' ', file=db)
        # np.set_printoptions(suppress=True)

        # print(' '.join(map(np.array_str, np.array(export_list_prey))), file=db)
        export_list_prey = []

        # print(' '.join(map(np.array_str, np.array(export_list_def))), file=db)
        for z in export_list_def:
            print(*z,sep=' ', file=db)
        export_list_def = []
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
