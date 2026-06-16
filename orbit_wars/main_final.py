import math
import collections

# ORBIT WARS - APEX V121 (The Overload - Aggressive Speed)
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

        # 1. Prediction
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

        # 2. Sequential Expansion (Blitz Priority)
        moves, spent, turn_launches = [], collections.defaultdict(float), collections.defaultdict(list)
        others = [p for p in planets if p[1] != p_idx]
        turns_left = 500 - obs.step

        for m in sorted(my, key=lambda x: x[5], reverse=True):
            m_id, mx, myy, m_s, m_p = m[0], m[2], m[3], m[5], m[6]

            # Fresh avail check
            all_m_ev = sorted(fleet_events[m_id] + turn_launches[m_id])
            _, _, min_s = simulate_planet(m, 45, all_m_ev)

            # v121 BLITZ BUFFER: 0.1 for first 3 planets
            buffer = 0.1 if len(my) < 4 else (1.1 if obs.step < 60 else 8.0)
            avail = min(m_s - buffer, (min_s - 0.2) if min_s > 0 else 0)
            if avail < 1: continue

            # Score targets for this source
            scored = []
            for t in others:
                dist = math.hypot(mx-t[2], myy-t[3])
                # Speed assume 30 for scoring
                eta = dist / get_fleet_speed(30)
                all_t_ev = sorted(fleet_events[t[0]] + turn_launches[t[0]])
                p_o, p_s, _ = simulate_planet(t, eta, all_t_ev)

                if p_o == p_idx: continue
                needed = int(p_s + 1)

                score = (t[6] * turns_left * 200 + 4000) / (dist + 5)
                if t[1] == -1:
                    score *= 15.0
                    # Step 0: Pure proximity
                    if obs.step == 0: score = 10**9 / (dist + 0.1)
                elif t[1] != -1:
                    if p_o != t[1]: score *= 3.0
                scored.append({'t': t, 'needed': needed, 'score': score, 'eta': eta})

            scored.sort(key=lambda x: x['score'], reverse=True)
            for item in scored:
                if avail < 1: break
                t_dat, needed = item['t'], item['needed']

                # Minimum send for speed in early game
                send = min(avail, max(needed, 18 if obs.step < 30 else 30))
                if t_dat[1] == -1 and avail < needed and obs.step > 20: continue

                send = int(send)
                if send < 1: continue

                speed = get_fleet_speed(send)
                f_pos = (t_dat[2], t_dat[3])
                for _ in range(4):
                    f_eta = math.hypot(mx-f_pos[0], myy-f_pos[1]) / speed
                    f_pos = get_predicted_pos(t_dat, f_eta, obs)

                dx, dy = f_pos[0]-mx, f_pos[1]-myy
                d2 = dx*dx + dy*dy
                if d2 > 1e-6:
                    tv = max(0, min(1, ((CX-mx)*dx+(CY-myy)*dy)/d2))
                    if ((mx+tv*dx-CX)**2 + (myy+tv*dy-CY)**2) < SUN_SAFE_SQ: continue

                moves.append([m_id, math.atan2(f_pos[1]-myy, f_pos[0]-mx), send])
                avail -= send
                turn_launches[t_dat[0]].append((f_eta, p_idx, send))
                # For neutrals, always break to avoid over-extension early
                if t_dat[1] == -1: break
        return moves
    except: return []
