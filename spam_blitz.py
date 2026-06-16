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
        if m_ships < 5: continue

        # Target top 10 closest planets
        others.sort(key=lambda o: (mx-o[2])**2 + (myy-o[3])**2)
        for t in others[:10]:
            angle = math.atan2(t[3]-myy, t[2]-mx)
            moves.append([m_id, angle, 1])
    return moves
