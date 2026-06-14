import math
import collections
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, ROTATION_RADIUS_LIMIT

CX, CY = 50.0, 50.0
SUN_SAFE = 10.8 # Safer buffer for high-speed fleets

def get_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_fleet_speed(ships):
    if ships <= 0: return 1.0
    return 1.0 + 5.0 * (math.log(max(1.1, ships)) / 6.9077) ** 1.5

def predict_pos(p, dt, obs):
    comet_ids = obs.get("comet_planet_ids", [])
    if p.id in comet_ids:
        for g in obs.get("comets", []):
            if p.id in g['planet_ids']:
                idx = g['planet_ids'].index(p.id)
                path = g['paths'][idx]
                f_idx = int(g['path_index'] + dt)
                if f_idx < len(path): return path[f_idx]
                return path[-1]
    d = math.hypot(p.x - CX, p.y - CY)
    if d > ROTATION_RADIUS_LIMIT: return (p.x, p.y)
    av = obs.get("angular_velocity", 0.0)
    angle = math.atan2(p.y - CY, p.x - CX) + av * dt
    return (CX + d * math.cos(angle), CY + d * math.sin(angle))

def is_path_safe(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    d2 = dx*dx + dy*dy
    if d2 < 1e-9: return True
    t = max(0, min(1, ((CX - p1[0]) * dx + (CY - p1[1]) * dy) / d2))
    return ((p1[0] + t * dx - CX)**2 + (p1[1] + t * dy - CY)**2) > SUN_SAFE**2

def agent(obs, config):
    try:
        player = obs.get("player", 0)
        planets = [Planet(*p) for p in obs.get("planets", [])]
        my_planets = [p for p in planets if p.owner == player]
        if not my_planets: return []

        raw_fleets = obs.get("fleets", [])
        incoming_net = collections.defaultdict(int)
        for f in raw_fleets:
            fl = Fleet(*f)
            best_p, min_diff = None, 0.15
            for p in planets:
                angle = math.atan2(p.y - fl.y, p.x - fl.x)
                diff = abs((angle - fl.angle + math.pi) % (2 * math.pi) - math.pi)
                if diff < min_diff: min_diff, best_p = diff, p
            if best_p:
                if fl.owner == player: incoming_net[best_p.id] += fl.ships
                else: incoming_net[best_p.id] -= fl.ships

        # Density Intelligence
        density = {}
        for p in planets:
            weights, values = [], []
            for other in planets:
                d = get_dist((p.x, p.y), (other.x, other.y))
                if d < 60:
                    w = math.exp(-d / 25.0)
                    val = other.ships
                    if other.owner == player: val *= 1.0
                    elif other.owner == -1: val = 0
                    else: val *= -2.0 # Heavy enemy penalty
                    weights.append(w)
                    values.append(val)
            density[p.id] = sum(w * v for w, v in zip(weights, values)) / sum(weights) if weights else 0

        moves, reserved = [], collections.defaultdict(int)

        # 1. High-Priority Defense
        threatened = [p for p in my_planets if p.ships + incoming_net[p.id] < 5]
        for t in threatened:
            needed = 12 - (t.ships + incoming_net[t.id])
            for m in sorted(my_planets, key=lambda p: get_dist((p.x, p.y), (t.x, t.y))):
                if m.id == t.id: continue
                buffer = 5 + int(m.production * 2)
                avail = m.ships - reserved[m.id] - buffer
                if avail > 0:
                    send = min(avail, needed)
                    d = get_dist((m.x, m.y), (t.x, t.y))
                    pos = predict_pos(t, d / 4.0, obs)
                    if is_path_safe((m.x, m.y), pos):
                        moves.append([m.id, math.atan2(pos[1] - m.y, pos[0] - m.x), int(send)])
                        reserved[m.id] += send
                        needed -= send
                        if needed <= 0: break

        # 2. Strategic Aggression
        targets = [p for p in planets if p.owner != player]
        def target_score(t):
            min_dist = min([get_dist((t.x, t.y), (m.x, m.y)) for m in my_planets])
            # Boost targets in high-dominance areas
            dom_boost = 1.0 / (1.0 + math.exp(-density[t.id] / 40.0))
            score = t.production * dom_boost / (t.ships + incoming_net[t.id] + 5) / (min_dist + 15)
            if t.owner == -1: score *= 1.8
            return -score

        targets.sort(key=target_score)

        for t in targets:
            if t.owner == -1 and incoming_net[t.id] > t.ships: continue

            # Coordination check: Can we take it?
            contribs = []
            for m in my_planets:
                buffer = 4 if density[m.id] > 0 else 10
                avail = m.ships - reserved[m.id] - buffer
                if avail > 0:
                    d = get_dist((m.x, m.y), (t.x, t.y))
                    eta = d / get_fleet_speed(avail)
                    contribs.append({'p': m, 'avail': avail, 'eta': eta})

            if not contribs: continue
            needed = t.ships - incoming_net[t.id] + 1
            if t.owner != -1: needed += 8

            if sum(c['avail'] for c in contribs) >= needed:
                contribs.sort(key=lambda x: x['eta'], reverse=True)
                max_eta = contribs[0]['eta']
                for c in contribs:
                    if needed <= 0: break
                    # Strike window: Launch if ETA is within a range to arrival synchronized
                    if c['eta'] > max_eta - 4:
                        m = c['p']
                        send = min(c['avail'], needed)
                        pos = predict_pos(t, c['eta'], obs)
                        if is_path_safe((m.x, m.y), pos):
                            moves.append([m.id, math.atan2(pos[1] - m.y, pos[0] - m.x), int(send)])
                            reserved[m.id] += send
                            needed -= send

        return moves
    except Exception:
        return []
