import math
import collections
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, ROTATION_RADIUS_LIMIT

CX, CY = 50.0, 50.0
SUN_SAFE = 10.2

def get_dist(p1_pos, p2_pos):
    return math.hypot(p1_pos[0] - p2_pos[0], p1_pos[1] - p2_pos[1])

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
    player = obs.get("player", 0)
    planets = [Planet(*p) for p in obs.get("planets", [])]
    my_planets = [p for p in planets if p.owner == player]
    if not my_planets: return []

    raw_fleets = obs.get("fleets", [])
    incoming_net = collections.defaultdict(int)
    for f in raw_fleets:
        fl = Fleet(*f)
        best_p, min_diff = None, 0.1
        for p in planets:
            angle = math.atan2(p.y - fl.y, p.x - fl.x)
            diff = abs((angle - fl.angle + math.pi) % (2 * math.pi) - math.pi)
            if diff < min_diff: min_diff, best_p = diff, p
        if best_p:
            if fl.owner == player: incoming_net[best_p.id] += fl.ships
            else: incoming_net[best_p.id] -= fl.ships

    # DENSITY INTELLIGENCE
    density = {}
    for p in planets:
        weights, values = [], []
        for other in planets:
            d = get_dist((p.x, p.y), (other.x, other.y))
            if d < 50:
                w = math.exp(-d / 20.0)
                val = other.ships
                if other.owner == player: val *= 1.0
                elif other.owner == -1: val = 0
                else: val *= -1.5
                weights.append(w)
                values.append(val)
        if weights:
            W = sum(weights)
            density[p.id] = sum(w * v for w, v in zip(weights, values)) / W
        else:
            density[p.id] = 0.0

    moves, reserved = [], collections.defaultdict(int)
    targets = [p for p in planets if p.owner != player]

    def target_score(t):
        dom = density[t.id]
        win_prob = 1.0 / (1.0 + math.exp(-dom / 50.0))
        return (t.owner != -1, -t.production * win_prob / (t.ships + 1))

    targets.sort(key=target_score)

    for t in targets:
        if t.owner == -1 and incoming_net[t.id] > t.ships: continue

        best_m, best_cost = None, float('inf')
        for m in my_planets:
            buffer = 2 if density[m.id] > 20 else 8
            avail = m.ships - reserved[m.id] - buffer
            if avail <= 0: continue

            d_init = get_dist((m.x, m.y), (t.x, t.y))
            pos = predict_pos(t, d_init / 3.0, obs)

            if is_path_safe((m.x, m.y), pos):
                speed = get_fleet_speed(avail)
                eta = get_dist((m.x, m.y), pos) / speed
                needed = t.ships - incoming_net[t.id] + 1
                if t.owner != -1: needed += int(t.production * eta) + 1

                if avail >= needed and needed > 0:
                    cost = eta / (t.production + 0.1)
                    if cost < best_cost:
                        best_cost, best_m = cost, (m.id, math.atan2(pos[1] - m.y, pos[0] - m.x), int(needed))

        if best_m:
            moves.append(list(best_m))
            reserved[best_m[0]] += best_m[2]

    return moves
