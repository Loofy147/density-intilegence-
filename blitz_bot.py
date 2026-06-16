import math

def agent(obs, config):
    p_idx = obs.player
    my = [p for p in obs.planets if p[1] == p_idx]
    if not my: return []
    others = [p for p in obs.planets if p[1] != p_idx]
    if not others: return []

    moves = []
    for m in my:
        m_id, mx, myy, m_ships = m[0], m[2], m[3], m[5]
        if m_ships < 2: continue

        # Target closest neutral/enemy
        others.sort(key=lambda o: (mx-o[2])**2 + (myy-o[3])**2)
        t = others[0]
        angle = math.atan2(t[3]-myy, t[2]-mx)
        # ALL IN
        moves.append([m_id, angle, int(m_ships - 1)])
    return moves
