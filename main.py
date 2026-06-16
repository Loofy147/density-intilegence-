import math
import collections

# ORBIT WARS - APEX V122 (The Conqueror - Sync & Steal)
CX, CY = 50.0, 50.0
SUN_SAFE_SQ = 11.0 ** 2

def get_fleet_speed(ships):
    s = max(1.1, float(ships))
    return 1.0 + 5.0 * (math.log(s) / 6.907755) ** 1.5

def get_predicted_pos(p_dat, eta, obs):
    t_id, tx, ty = p_dat[0], p_dat[2], p_dat[3]
    for g in obs.get("comets", []):
        if t_id in g.get('planet_ids', []):
            idx = g['planet_ids'].index(t_id)
            path = g['paths'][idx]
            f_idx = int(round(g['path_index'] + eta))
            if f_idx < len(path): return path[f_idx]
            return path[-1]
    av = obs.get("angular_velocity", 0.0)
    d = math.hypot(tx-CX, ty-CY)
    if d < 1e-6 or d > 55.0: return (tx, ty)
    cur_ang = math.atan2(ty-CY, tx-CX)
    res_ang = cur_ang + av * eta
    return (CX + d * math.cos(res_ang), CY + d * math.sin(res_ang))

def simulate_planet(p_dat, target_eta, events):
    curr_o, curr_s, curr_t = p_dat[1], float(p_dat[5]), 0
    p_p = float(p_dat[6])
    min_s = curr_s if curr_o != -1 else -curr_s
    for e_eta, e_o, e_s in sorted(events):
        if e_eta > target_eta: break
        dt = e_eta - curr_t
        if curr_o != -1: curr_s += p_p * dt
        curr_t = e_eta
        if e_o == curr_o: curr_s += e_s
        else:
            if e_s > curr_s + 1e-5:
                curr_s, curr_o = e_s - curr_s, e_o
            elif abs(e_s - curr_s) <= 1e-5:
                curr_s, curr_o = 0, -1
            else: curr_s -= e_s
        if curr_o == p_dat[1] and curr_o != -1:
            if curr_s < min_s: min_s = curr_s
        elif p_dat[1] != -1:
            min_s = -curr_s
    dt = target_eta - curr_t
    if curr_o != -1: curr_s += p_p * dt
    return curr_o, curr_s, min_s

def agent(obs, config):
    try:
        p_idx, planets = obs.player, obs.planets
        id_map = {p[0]: p for p in planets}
        my = [p for p in planets if p[1] == p_idx]
        if not my: return []

        # 1. Prediction Mapping
        fleet_events = collections.defaultdict(list)
        for f in obs.get("fleets", []):
            f_o, fx, fy, f_ang, f_s = f[1], f[2], f[3], f[4], f[6]
            tid, min_d = None, 0.45
            for p in planets:
                ang = math.atan2(p[3]-fy, p[2]-fx)
                diff = abs((ang-f_ang+math.pi)%(2*math.pi)-math.pi)
                if diff < min_d: min_d, tid = diff, p[0]
            if tid is not None:
                dist = math.hypot(id_map[tid][2]-fx, id_map[tid][3]-fy)
                fleet_events[tid].append((dist/get_fleet_speed(f_s), f_o, f_s))

        # 2. Mission Scoring
        missions, turns_left = [], 500 - obs.step
        for m in my:
            m_id, mx, myy, m_s, m_p = m[0], m[2], m[3], m[5], m[6]
            _, _, min_s = simulate_planet(m, 45, fleet_events[m_id])

            # v122 aggression logic
            buffer = 0.1 if obs.step < 50 else 8.0
            m_avail = min(m_s - buffer, (min_s - 0.1) if min_s > 0 else (m_s - 0.5))
            if m_avail < 1: continue

            for t in planets:
                if t[0] == m_id: continue
                dist = math.hypot(mx-t[2], myy-t[3])
                eta = dist / get_fleet_speed(25)

                # Joint Strike / Arrival Sync
                is_sync = any(e_o == p_idx and abs(e_eta - eta) < 8 for e_eta, e_o, e_s in fleet_events[t[0]])

                p_o, p_s, min_s_t = simulate_planet(t, eta, fleet_events[t[0]])
                if p_o == p_idx:
                    if min_s_t < 15.0:
                        missions.append({'src':m, 'dst':t, 'needed':int(40-min_s_t), 'score':5000000/(dist+1), 'type':'def', 'eta':eta})
                    continue

                needed = int(p_s + 1)
                # v122 refined scoring: production weighted by match stage
                prod_weight = 300 if obs.step < 100 else 600
                score = (t[6] * turns_left * prod_weight + 10000) / (dist + 5)
                if is_sync: score *= 2.0

                if t[1] == -1:
                    score *= 20.0
                    # Steal-Timing: arrive immediately after enemy
                    enemy_arr = [e_eta for e_eta, e_o, e_s in fleet_events[t[0]] if e_o != p_idx and e_o != -1]
                    if enemy_arr and min(enemy_arr) < eta: score *= 4.0

                    if obs.step == 0: score = 10**9 / (dist + 0.1)
                    if m_avail < needed and obs.step > 25: score *= 0.00001
                elif t[1] != -1:
                    if p_o != t[1]: score *= 4.0 # Active conflict

                missions.append({'src':m, 'dst':t, 'needed':needed, 'score':score, 'type':'atk', 'eta':eta})

        missions.sort(key=lambda x: x['score'], reverse=True)

        # 3. Execution
        moves, spent, covered = [], collections.defaultdict(float), collections.defaultdict(float)
        turn_launches = collections.defaultdict(list)
        for mis in missions:
            if len(moves) >= 80: break
            m, t = mis['src'], mis['dst']
            all_m_ev = sorted(fleet_events[m[0]] + turn_launches[m[0]])
            _, _, min_s = simulate_planet(m, 45, all_m_ev)
            buffer = 0.1 if obs.step < 50 else 8.0
            m_avail = min(m[5] - spent[m[0]] - buffer, (min_s - spent[m[0]] - 0.2) if min_s > 0 else (m[5] - spent[m[0]] - 0.5))
            if m_avail < 1: continue

            all_t_ev = sorted(fleet_events[t[0]] + turn_launches[t[0]])
            speed_est = get_fleet_speed(min(m_avail, 25))
            eta_est = math.hypot(m[2]-t[2], m[3]-t[3]) / speed_est
            p_o, p_s, _ = simulate_planet(t, eta_est, all_t_ev)
            if p_o == p_idx: continue

            needed = int(p_s + 1)
            if mis['type'] == 'atk':
                if t[1] == -1 and m_avail < needed and obs.step > 30: continue
                send = min(m_avail, max(needed, 18 if obs.step < 60 else 45))
            else:
                send = min(m_avail, mis['needed'] - covered[t[0]])

            send = int(send)
            if send < 2: continue
            speed = get_fleet_speed(send)
            f_pos = (t[2], t[3])
            for _ in range(4):
                f_eta = math.hypot(m[2]-f_pos[0], m[3]-f_pos[1]) / speed
                f_pos = get_predicted_pos(t, f_eta, obs)
            dx, dy = f_pos[0]-m[2], f_pos[1]-m[3]
            d2 = dx*dx + dy*dy
            if d2 > 0:
                tv = max(0, min(1, ((CX-m[2])*dx+(CY-m[3])*dy)/d2))
                if ((m[2]+tv*dx-CX)**2 + (m[3]+tv*dy-CY)**2) < SUN_SAFE_SQ: continue

            moves.append([m[0], math.atan2(f_pos[1]-m[3], f_pos[0]-m[2]), send])
            spent[m[0]], covered[t[0]] = spent[m[0]]+send, covered[t[0]]+send
            if t[1] == -1: break
        return moves
    except: return []
