import math
import collections

# ORBIT WARS - APEX V78
CX, CY = 50.0, 50.0
SUN_SAFE_SQ = 10.14 ** 2

def get_fleet_speed(ships):
    s = max(1.1, float(ships))
    return 1.0 + 5.0 * (math.log(s) / 6.907755) ** 1.5

def get_predicted_pos(t_id, t_pos, eta, obs):
    # 1. Comets
    for g in obs.get("comets", []):
        if t_id in g['planet_ids']:
            idx = g['planet_ids'].index(t_id)
            path = g['paths'][idx]
            # path_index is current index
            f_idx = int(g['path_index'] + eta)
            if f_idx < len(path): return path[f_idx]
            return path[-1]

    # 2. Orbits
    av = obs.get("angular_velocity", 0.0)
    d = math.hypot(t_pos[0]-CX, t_pos[1]-CY)
    if d < 1e-3: return t_pos
    if d > 40.0: return t_pos # Not orbiting (shouldn't happen for non-comets in most maps)

    cur_ang = math.atan2(t_pos[1]-CY, t_pos[0]-CX)
    res_ang = cur_ang + av * eta
    return (CX + d * math.cos(res_ang), CY + d * math.sin(res_ang))

def agent(obs, config):
    try:
        p_idx = obs.player
        planets = obs.planets
        my = [p for p in planets if p[1] == p_idx]
        if not my: return []

        # 1. Fleet Tracking
        net_impact = collections.defaultdict(int)
        for f in obs.get("fleets", []):
            f_owner, f_pos, f_angle, f_ships = f[1], (f[2], f[3]), f[4], f[6]
            target_id, min_diff = None, 0.3
            for p in planets:
                ang = math.atan2(p[3]-f_pos[1], p[2]-f_pos[0])
                diff = abs((ang - f_angle + math.pi) % (2*math.pi) - math.pi)
                if diff < min_diff: min_diff, target_id = diff, p[0]
            if target_id is not None:
                if f_owner == p_idx: net_impact[target_id] -= f_ships
                else: net_impact[target_id] += f_ships

        moves = []
        for m in sorted(my, key=lambda x: x[5], reverse=True):
            m_id, m_pos, m_ships = m[0], (m[2], m[3]), m[5]

            # Defense: Minimum buffer
            # In early game (few planets), be more aggressive.
            buffer = 2.0 if len(my) < 3 else 10.0
            danger = max(0, net_impact[m_id])
            m_avail = m_ships - (buffer + danger)
            if m_avail < 2: continue

            # 2. Target Ranking
            targets = []
            for t in planets:
                if t[1] == p_idx: continue
                t_id, t_owner, t_pos, t_ships, t_prod = t[0], t[1], (t[2], t[3]), t[5], t[6]

                dist = math.hypot(m_pos[0]-t_pos[0], m_pos[1]-t_pos[1])
                resistance = t_ships + net_impact[t_id]
                if t_owner != -1:
                    resistance += (t_prod * (dist / 16.0))

                needed = int(resistance + 1)

                # Proximity is prioritized for early expansion
                score = (t_prod + 5) / (dist + 10)
                if t_owner == -1:
                    score *= 10.0
                    if m_avail >= needed: score *= 2.0
                    else: score *= 0.05 # Don't send if we can't take

                targets.append({'id': t_id, 'pos': t_pos, 'needed': needed, 'score': score, 'owner': t_owner})

            targets.sort(key=lambda x: x['score'], reverse=True)

            # 3. Launch
            for t in targets:
                if m_avail < 2: break

                needed = t['needed']
                if t['owner'] == -1 and m_avail < needed: continue

                # Send more for speed if we have it
                send = min(m_avail, max(needed, 15))
                if t['owner'] != -1: send = min(m_avail, max(send, 25))

                send = int(send)
                if send < 1: continue

                # Interception
                speed = get_fleet_speed(send)
                f_target = t['pos']
                for _ in range(3):
                    eta = math.hypot(m_pos[0]-f_target[0], m_pos[1]-f_target[1]) / speed
                    f_target = get_predicted_pos(t['id'], t['pos'], eta, obs)

                # Sun
                dx, dy = f_target[0]-m_pos[0], f_target[1]-m_pos[1]
                d2 = dx*dx + dy*dy
                safe = True
                if d2 > 0:
                    tv = max(0, min(1, ((CX-m_pos[0])*dx + (CY-m_pos[1])*dy) / d2))
                    if ((m_pos[0]+tv*dx-CX)**2 + (m_pos[1]+tv*dy-CY)**2) < SUN_SAFE_SQ:
                        safe = False

                if safe:
                    moves.append([m_id, math.atan2(f_target[1]-m_pos[1], f_target[0]-m_pos[0]), send])
                    m_avail -= send
                    net_impact[t['id']] -= send
                    if t['owner'] == -1: break # One neutral per source
        return moves
    except: return []
