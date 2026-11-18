from ortools.linear_solver import pywraplp

def optimize_order_quantity(demand,on_hand,holding_cost=1.0,stockout_cost=8.0,pack_size=1,budget=None,unit_cost=1.0,lead_time_demand=None):
    if min(demand,on_hand,holding_cost,stockout_cost,unit_cost)<0 or pack_size<1: raise ValueError("Inputs must be non-negative; pack_size >= 1.")
    target=max(0.0,float(lead_time_demand if lead_time_demand is not None else demand))
    solver=pywraplp.Solver.CreateSolver("CBC")
    if solver is None: raise RuntimeError("OR-Tools CBC solver unavailable.")
    max_order=max(int(target*3+on_hand+pack_size),pack_size)
    packs=solver.IntVar(0,max_order//pack_size+1,"packs")
    order=solver.IntVar(0,max_order,"order")
    ending=solver.NumVar(0,solver.infinity(),"ending")
    shortage=solver.NumVar(0,solver.infinity(),"shortage")
    solver.Add(order==packs*pack_size)
    solver.Add(ending>=on_hand+order-target)
    solver.Add(shortage>=target-on_hand-order)
    if budget is not None: solver.Add(order*unit_cost<=float(budget))
    solver.Minimize(holding_cost*ending+stockout_cost*shortage+1e-5*order)
    status=solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL,pywraplp.Solver.FEASIBLE):
        return {"status":"infeasible","order_quantity":0,"ending_inventory":0,"expected_shortage":0}
    return {"status":"optimal" if status==pywraplp.Solver.OPTIMAL else "feasible",
            "order_quantity":int(round(order.solution_value())),"ending_inventory":round(ending.solution_value(),2),
            "expected_shortage":round(shortage.solution_value(),2),"objective_cost":round(solver.Objective().Value(),2)}
