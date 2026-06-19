import math
import collections

# ORBIT WARS - APEX V480 (The Alpha Conqueror)
CX, CY = 50.0, 50.0
SUN_SAFE_SQ = 10.15 ** 2

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
    return (CX + d * math.cos(cur_ang + av*eta), CY + d * math.sin(cur_ang + av*eta))

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
        my = [p for p in planets if p[1] == p_idx]
        if not my: return []
        others = [p for p in planets if p[1] != p_idx]

        # --- PHASE 1: TURN 0 ALL-IN (Counter-Blitz) ---
        if obs.step == 0:
            m = my[0]; mx, myy, m_s = m[2], m[3], m[5]
            # Find closest neutral. MUST match Blitz bot speed.
            neutrals = sorted([n for n in others if n[1] == -1], key=lambda n: (mx-n[2])**2 + (myy-n[3])**2)
            t = neutrals[0]
            send = int(m_s - 1)
            eta = math.hypot(mx-t[2], myy-t[3]) / get_fleet_speed(send)
            f_pos = get_predicted_pos(t, eta, obs)
            return [[m[0], math.atan2(f_pos[1]-myy, f_pos[0]-mx), send]]

        # --- PHASE 2: STRATEGIC RECURSIVE ---
        fleet_events = collections.defaultdict(list)
        for f in obs.get("fleets", []):
            f_o, fx, fy, f_ang, f_s = f[1], f[2], f[3], f[4], f[6]
            tid, min_d = None, 0.4
            for p in planets:
                ang = math.atan2(p[3]-fy, p[2]-fx)
                if abs((ang-f_ang+math.pi)%(2*math.pi)-math.pi) < min_d: tid = p[0]; break
            if tid is not None:
                fleet_events[tid].append((math.hypot(planets[tid][2]-fx, planets[tid][3]-fy)/get_fleet_speed(f_s), f_o, f_s))

        moves, spent = [], collections.defaultdict(float)

        # Priority 1: Instant Defense
        for m in sorted(my, key=lambda x: x[5]):
            m_id = m[0]
            _, _, min_s = simulate_planet(m, 35, fleet_events[m_id])
            if min_s < 1.0:
                needed = abs(min_s) + 5
                helpers = sorted([p for p in my if p[0] != m_id], key=lambda p: math.hypot(p[2]-m[2], p[3]-m[3]))
                for h in helpers:
                    h_avail = h[5] - spent[h[0]] - 2.0
                    if h_avail > 2:
                        send = min(h_avail, needed)
                        eta = math.hypot(h[2]-m[2], h[3]-m[3]) / get_fleet_speed(send)
                        f_pos = get_predicted_pos(m, eta, obs)
                        moves.append([h[0], math.atan2(f_pos[1]-h[3], f_pos[0]-h[2]), int(send)])
                        spent[h[0]] += send; needed -= send
                        if needed <= 0: break

        # Priority 2: Sequential Expansion (Concentrated Force)
        for m in sorted(my, key=lambda x: x[5], reverse=True):
            m_id, mx, myy, m_s = m[0], m[2], m[3], m[5]
            buffer = 1.1 if obs.step < 60 else 15.0
            avail = m_s - spent[m_id] - buffer
            if avail < 10: continue

            candidates = []
            for t in others:
                d = math.hypot(mx-t[2], myy-t[3])
                p_o, p_s, _ = simulate_planet(t, 40, fleet_events[t[0]])
                if p_o == p_idx: continue
                score = (t[6] * 1000 + 500) / (d + 1)
                if t[1] == -1: score *= 10.0
                candidates.append((score, t, d, p_s))

            candidates.sort(key=lambda x: x[0], reverse=True)
            for score, t, d, p_s in candidates:
                send = avail
                eta = d / get_fleet_speed(send)
                f_pos = get_predicted_pos(t, eta, obs)

                # Basic sun safety
                dx, dy = f_pos[0]-mx, f_pos[1]-myy; d2 = dx*dx + dy*dy
                if d2 > 0:
                    tv = max(0, min(1, ((50-mx)*dx+(50-myy)*dy)/d2))
                    if ((mx+tv*dx-50)**2 + (myy+tv*dy-50)**2) < SUN_SAFE_SQ: continue

                moves.append([m_id, math.atan2(f_pos[1]-myy, f_pos[0]-mx), int(send)])
                spent[m_id] += send
                break

        return moves
    except: return []
