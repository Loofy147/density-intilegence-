import math
import collections

def get_fleet_speed(ships):
    s = max(1.1, float(ships))
    return 1.0 + 5.0 * (math.log(s) / 6.907755) ** 1.5

def simulate_planet(p_id, p_owner, p_ships, p_prod, target_eta, events):
    curr_ships = float(p_ships)
    curr_owner = p_owner
    curr_time = 0
    min_ships = curr_ships
    for e_eta, e_owner, e_ships in sorted(events):
        if e_eta > target_eta: break
        dt = e_eta - curr_time
        if curr_owner != -1: curr_ships += p_prod * dt
        curr_time = e_eta
        if e_owner == curr_owner: curr_ships += e_ships
        else:
            curr_ships -= e_ships
            if curr_ships < 0:
                curr_ships = abs(curr_ships)
                curr_owner = e_owner
        if curr_owner == p_owner: min_ships = min(min_ships, curr_ships)
        else: min_ships = -curr_ships
    dt = target_eta - curr_time
    if curr_owner != -1: curr_ships += p_prod * dt
    return curr_owner, curr_ships, min_ships

def test_simulation():
    # Case 1: Neutral planet with incoming ally fleet
    # p: id=0, owner=-1, ships=10, prod=2
    # fleet: eta=5, owner=0, ships=20
    # target_eta = 10
    owner, ships, min_s = simulate_planet(0, -1, 10, 2, 10, [(5, 0, 20)])
    assert owner == 0, f"Expected owner 0, got {owner}"
    # At t=5: 10 ships neutral. Fleet arrives: 20 ships vs 10 neutral -> owner 0, ships 10.
    # From t=5 to t=10: owner 0 produces 2*5=10 ships. Total 20.
    assert ships == 20, f"Expected ships 20, got {ships}"

    # Case 2: Enemy planet production
    # p: owner=1, ships=10, prod=1. target=10.
    owner, ships, _ = simulate_planet(1, 1, 10, 1, 10, [])
    assert owner == 1
    assert ships == 20 # 10 + 1*10

    # Case 3: Defense simulation (min_ships)
    # p: owner=0, ships=50, prod=1. Enemy fleet (1, 60) arrives at t=10.
    owner, ships, min_s = simulate_planet(2, 0, 50, 1, 20, [(10, 1, 60)])
    # At t=10: 50 + 1*10 = 60. Enemy 60 vs My 60.
    # Logic curr_ships -= e_ships -> 0. owner remains 0 (because curr_ships not < 0)
    # From t=10 to t=20: 0 + 1*10 = 10.
    assert owner == 0
    assert ships == 10
    assert min_s == 0

    print("Logic validation tests passed!")

if __name__ == "__main__":
    test_simulation()
