import math
import collections

# ORBIT WARS - APEX V79.2 (Optimized Time-Aware)
CX, CY = 50.0, 50.0
SUN_SAFE_SQ = 10.14 ** 2

def get_fleet_speed(ships):
    s = max(1.1, float(ships))
    return 1.0 + 5.0 * (math.log(s) / 6.907755) ** 1.5

def get_predicted_pos(t_id, t_pos, eta, obs):
    # Comets
    for g in obs.get("comets", []):
        if t_id in g.get('planet_ids', []):
            idx = g['planet_ids'].index(t_id)
            path = g['paths'][idx]
            f_idx = int(g['path_index'] + eta)
            if f_idx < len(path): return path[f_idx]
            return path[-1]
    # Orbits
    av = obs.get("angular_velocity", 0.0)
    d2 = (t_pos[0]-CX)**2 + (t_pos[1]-CY)**2
    if d2 < 1e-6 or d2 > 1600.0: return t_pos
    d = d2 ** 0.5
    cur_ang = math.atan2(t_pos[1]-CY, t_pos[0]-CX)
    res_ang = cur_ang + av * eta
    return (CX + d * math.cos(res_ang), CY + d * math.sin(res_ang))

def simulate_planet_fast(p_owner, p_ships, p_prod, target_eta, sorted_events):
    """Predicts planet state at target_eta using pre-sorted events."""
    curr_ships = float(p_ships)
    curr_owner = p_owner
    curr_time = 0
    min_ships = curr_ships

    for e_eta, e_owner, e_ships in sorted_events:
        if e_eta > target_eta: break

        # Production
        dt = e_eta - curr_time
        if curr_owner != -1:
            curr_ships += p_prod * dt
        curr_time = e_eta

        # Fleet Arrival
        if e_owner == curr_owner:
            curr_ships += e_ships
        else:
            curr_ships -= e_ships
            if curr_ships < 0:
                curr_ships = abs(curr_ships)
                curr_owner = e_owner

        if curr_owner == p_owner:
            if curr_ships < min_ships: min_ships = curr_ships
        else:
            min_ships = -curr_ships # Lost it

    # Final advance
    dt = target_eta - curr_time
    if curr_owner != -1:
        curr_ships += p_prod * dt

    return curr_owner, curr_ships, min_ships

def agent(obs, config):
    try:
        p_idx = obs.player
        planets = obs.planets
        my = [p for p in planets if p[1] == p_idx]
        if not my: return []

        # 1. Pre-process Fleet Events
        fleet_events = collections.defaultdict(list)
        for f in obs.get("fleets", []):
            f_owner, fx, fy, f_angle, f_ships = f[1], f[2], f[3], f[4], f[6]
            target_id, min_diff = None, 0.35
            best_dist = 0
            for p in planets:
                ang = math.atan2(p[3]-fy, p[2]-fx)
                diff = abs((ang - f_angle + math.pi) % (2*math.pi) - math.pi)
                if diff < min_diff:
                    min_diff, target_id = diff, p[0]
                    best_dist = math.hypot(p[2]-fx, p[3]-fy)

            if target_id is not None:
                speed = get_fleet_speed(f_ships)
                fleet_events[target_id].append((best_dist / speed, f_owner, f_ships))

        for pid in fleet_events:
            fleet_events[pid].sort()

        moves = []
        # Local tracking within turn
        turn_launches = collections.defaultdict(list)

        for m in sorted(my, key=lambda x: x[5], reverse=True):
            m_id, m_owner, mx, my_pos, _, m_ships, m_prod = m

            # Combine pre-existing fleet events and launches from this turn
            m_events = fleet_events[m_id] + turn_launches[m_id]
            _, _, min_s = simulate_planet_fast(m_owner, m_ships, m_prod, 40, sorted(m_events))

            # Conservative buffer: keep more if we have many planets
            buffer = 10.0 if len(my) > 3 else 3.0
            m_avail = min(m_ships - buffer, min_s - 1.0 if min_s > 0 else 0)
            if m_avail < 5: continue

            # 2. Rank Targets
            targets = []
            for t in planets:
                if t[1] == p_idx: continue
                t_id, t_owner, tx, ty, _, t_ships, t_prod = t

                dist = math.hypot(mx-tx, my_pos-ty)
                est_eta = dist / get_fleet_speed(25)

                t_events = fleet_events[t_id] + turn_launches[t_id]
                pred_owner, pred_ships, _ = simulate_planet_fast(t_owner, t_ships, t_prod, est_eta, sorted(t_events))

                if pred_owner == p_idx: continue

                needed = int(pred_ships + 1)
                score = (t_prod + 5) / (dist + 15)
                if t_owner == -1:
                    score *= 12.0
                    if m_avail < needed: score *= 0.01

                targets.append({'t': t, 'needed': needed, 'score': score, 'dist': dist})

            targets.sort(key=lambda x: x['score'], reverse=True)

            # 3. Multi-mission
            for item in targets:
                if m_avail < 5: break
                t_dat = item['t']
                needed = item['needed']

                if t_dat[1] == -1 and m_avail < needed: continue

                send = min(m_avail, max(needed, 20))
                if t_dat[1] != -1: send = min(m_avail, max(send, 45))

                send = int(send)
                if send < 5: continue

                speed = get_fleet_speed(send)
                f_pos = (t_dat[2], t_dat[3])
                for _ in range(3):
                    eta = math.hypot(mx-f_pos[0], my_pos-f_pos[1]) / speed
                    f_pos = get_predicted_pos(t_dat[0], (t_dat[2], t_dat[3]), eta, obs)

                dx, dy = f_pos[0]-mx, f_pos[1]-my_pos
                d2 = dx*dx + dy*dy
                safe = True
                if d2 > 0:
                    tv = max(0, min(1, ((CX-mx)*dx + (CY-my_pos)*dy) / d2))
                    if ((mx+tv*dx-CX)**2 + (my_pos+tv*dy-CY)**2) < SUN_SAFE_SQ:
                        safe = False

                if safe:
                    moves.append([m_id, math.atan2(f_pos[1]-my_pos, f_pos[0]-mx), send])
                    m_avail -= send
                    # Record for other sources in this turn
                    turn_launches[t_dat[0]].append((eta, p_idx, send))
                    if t_dat[1] == -1: break
        return moves
    except: return []
