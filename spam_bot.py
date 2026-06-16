import math
import collections

def agent(obs, config):
    try:
        p_idx = obs.player
        my = [p for p in obs.planets if p[1] == p_idx]
        if not my: return []

        others = [p for p in obs.planets if p[1] != p_idx]
        if not others: return []

        moves = []
        for m in my:
            # Spam: send 1 ship to EVERY other planet every turn if we have > 2 ships
            m_id, mx, my_pos, m_ships = m[0], m[2], m[3], m[5]
            if m_ships < 5: continue

            for t in others[:3]: # Limit to top 3 closest to avoid move limit issues
                angle = math.atan2(t[3]-my_pos, t[2]-mx)
                moves.append([m_id, angle, 1])
        return moves
    except: return []
