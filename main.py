import math
import collections
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, ROTATION_RADIUS_LIMIT

# PHYSICS CONSTANTS
CX, CY = 50.0, 50.0
SUN_SAFE = 10.12 # Very aggressive path skimming

def get_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_fleet_speed(ships):
    if ships <= 1.1: return 1.0
    # Speed formula: 1 + 5 * (log(max(1.1, ships)) / log(1000))^1.5
    return 1.0 + 5.0 * (math.log(ships) / 6.9077) ** 1.5

def predict_pos(p, dt, av, obs):
    # COMETS
    comet_ids = obs.get("comet_planet_ids", [])
    if p.id in comet_ids:
        for g in obs.get("comets", []):
            if p.id in g['planet_ids']:
                idx = g['planet_ids'].index(p.id)
                path = g['paths'][idx]
                f_idx = int(g['path_index'] + dt)
                if f_idx < len(path): return path[f_idx]
                return path[-1]
    # ORBIT
    d = math.hypot(p.x - CX, p.y - CY)
    if d > ROTATION_RADIUS_LIMIT: return (p.x, p.y)
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

        av = obs.get("angular_velocity", 0.0)

        # 1. NET SHIP TRACKING (Arrival prediction)
        net = collections.defaultdict(int)
        for p in planets:
            net[p.id] = p.ships if p.owner != player else -p.ships

        for f in obs.get("fleets", []):
            fl = Fleet(*f)
            # Find destination (closest planet by angle)
            best_p, min_diff = None, 0.2
            for p in planets:
                ang = math.atan2(p.y - fl.y, p.x - fl.x)
                diff = abs((ang - fl.angle + math.pi) % (2 * math.pi) - math.pi)
                if diff < min_diff: min_diff, best_p = diff, p

            if best_p:
                if fl.owner == player:
                    if best_p.owner != player: net[best_p.id] -= fl.ships
                else:
                    net[best_p.id] += fl.ships

        # 2. LOCAL DENSITY (Contextual intelligence)
        density = {}
        for p in planets:
            w, v = [], []
            for o in planets:
                d = get_dist((p.x, p.y), (o.x, o.y))
                if d < 60:
                    weight = math.exp(-d / 25.0)
                    val = o.ships if o.owner == player else (-1.8 * o.ships if o.owner != -1 else 0)
                    w.append(weight)
                    v.append(val)
            density[p.id] = sum(wi * vi for wi, vi in zip(w, v)) / sum(w) if w else 0

        moves, reserved = [], collections.defaultdict(int)

        # 3. HYPER-AGGRESSIVE EXPANSION & INTERCEPTION
        others = [p for p in planets if p.owner != player]

        for m in sorted(my_planets, key=lambda x: x.ships, reverse=True):
            if m.ships < 2: continue

            # Sort targets for THIS planet based on predicted value
            def target_score(o):
                dist = get_dist((m.x, m.y), (o.x, o.y))
                resistance = max(0, net[o.id])
                # Value = Production / (Time + Resistance)
                score = o.production / (resistance + 1) / (dist + 20)
                if o.owner == -1: score *= 10.0 # Extreme neutral grab
                return score

            scored_targets = sorted(others, key=target_score, reverse=True)

            m_avail = m.ships - 1
            for t in scored_targets:
                if m_avail <= 0: break

                res = max(0, net[t.id])
                if t.owner == -1 and net[t.id] < 0: continue # Already winning

                needed = res + 1
                if t.owner != -1: needed += 5 # Margin

                send = min(m_avail, needed)
                # SCALE: Send slightly more if we have a lot, to move faster
                if t.owner == -1 and m_avail > 15: send = max(send, 12)
                send = min(send, m_avail)

                if send > 0:
                    # ITERATIVE CURVATURE INTERCEPTION
                    dist = get_dist((m.x, m.y), (t.x, t.y))
                    speed = get_fleet_speed(send)
                    eta = dist / speed
                    pos = predict_pos(t, eta, av, obs)
                    # Refine twice
                    for _ in range(2):
                        eta = get_dist((m.x, m.y), pos) / speed
                        pos = predict_pos(t, eta, av, obs)

                    if is_path_safe((m.x, m.y), pos):
                        moves.append([m.id, math.atan2(pos[1] - m.y, pos[0] - m.x), int(send)])
                        m_avail -= send
                        reserved[m.id] += send
                        net[t.id] -= send
                        if t.owner == -1 and net[t.id] < 0:
                            others = [o for o in others if o.id != t.id]
                        break # One mission per planet to maximize breadth

        # 4. EMERGENCY DEFENSE
        for p in my_planets:
            if net[p.id] > 0:
                needed = net[p.id] + 2
                for h in sorted(my_planets, key=lambda x: get_dist((x.x, x.y), (p.x, p.y))):
                    if h.id == p.id: continue
                    avail = h.ships - reserved[h.id] - 1
                    if avail > 0:
                        send = min(avail, needed)
                        moves.append([h.id, math.atan2(p.y - h.y, p.x - h.x), int(send)])
                        reserved[h.id] += send
                        net[p.id] -= send
                        needed -= send
                        if net[p.id] <= 0: break

        return moves
    except Exception: return []
