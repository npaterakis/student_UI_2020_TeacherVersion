import pandas, numpy
from pyomo.environ import *

def stochasticRisk(DApriceFC, WGScen, IMplus, IMminus, probs, alpha, beta):
    model = ConcreteModel()

    #Define sets
    model.T = Set(ordered=True, initialize= DApriceFC.index)
    model.S = Set(ordered=True, initialize= WGScen.index)

    #Define and initialize parameters
    model.Pcap     = Param(within=NonNegativeReals, initialize= 25) #wind-farm installed capacity in MW !!! DO NOT CHANGE!
    model.P_W_DA   = Param(model.T, model.S, within=NonNegativeReals, mutable=True) #wind power scenarios in MW
    model.Price_DA = Param(model.T, within=NonNegativeReals, mutable=True, initialize=DApriceFC['DAP'].to_dict()) #day-ahead market price in Euros/MWh
    model.dplus    = Param(model.T, model.S, within=Reals, mutable=True)  # modifier positive imbalance
    model.dminus   = Param(model.T, model.S, within=Reals, mutable=True)  # modifier negative imbalance
    model.prob     = Param(model.S, within=NonNegativeReals, initialize=probs['prob'].to_dict())  # probabilities of scenarios

    model.alpha    = Param(within=NonNegativeReals, initialize=alpha)
    model.beta     = Param(within=NonNegativeReals, initialize=beta)

    for t in model.T:
        for s in model.S:
            model.P_W_DA[t,s] = WGScen.loc[s,t]
            model.dplus[t,s] = IMplus.loc[s,t]
            model.dminus[t,s] = IMminus.loc[s,t]

    #Define decision variables
    model.P_DA        = Var(model.T, within=NonNegativeReals) #offer DA (MWh)
    model.Delta       = Var(model.T, model.S, within=Reals) #imbalance volume (MWh)
    model.Delta_plus  = Var(model.T, model.S, within= NonNegativeReals) #positive imbalance volume (MWh)
    model.Delta_minus = Var(model.T, model.S, within= NonNegativeReals) #negative imbalance volume (MWh)
    model.zeta        = Var(within=Reals) #auxilliary variable (equals to VaR at optimality)
    model.eta         = Var(model.S, within=NonNegativeReals) #auxilliary variable for CVaR calculation
    model.cvar        = Var(within=Reals) #CVaR (Euros)
    model.EP          = Var(within=Reals) #Expected profit (Euros)
    model.scenario_cost = Var(model.S, within=Reals)
    model.EIm_hourly = Var(model.T, within=Reals)
    #Define problem objective and constraints

    def obj(model):
        return  (1-model.beta)*model.EP + model.beta*model.cvar

    def con_EP(model): #not an actual constraint, it defines expected profit
        #return model.EP == sum(sum(model.prob[s] * (model.Price_DA[t,s]*model.P_DA[t] + model.Price_DA[t,s]*model.dplus[t,s]*model.Delta_plus[t,s] - model.Price_DA[t,s]*model.dminus[t,s]*model.Delta_minus[t,s]) for t in model.T) for s in model.S)
        return model.EP == sum(sum(model.prob[s] * (model.Price_DA[t]*model.P_DA[t] + model.Price_DA[t]*model.dplus[t,s]*model.Delta_plus[t,s] - model.Price_DA[t]*model.dminus[t,s]*model.Delta_minus[t,s]) for s in model.S) for t in model.T)

    def scenarioCost(model,s):
        return model.scenario_cost[s] == sum(model.Price_DA[t]*model.P_DA[t] + model.Price_DA[t]*model.dplus[t,s]*model.Delta_plus[t,s] - model.Price_DA[t]*model.dminus[t,s]*model.Delta_minus[t,s] for t in model.T)

    def con_cvar1(model, s):
        return - sum(model.Price_DA[t]*model.P_DA[t] + model.Price_DA[t]*model.dplus[t,s]*model.Delta_plus[t,s] - model.Price_DA[t]*model.dminus[t,s]*model.Delta_minus[t,s] for t in model.T)+ model.zeta - model.eta[s] <= 0

    def con_cvar2(model): #Definition of CVaR
        return model.cvar == model.zeta - (1/(1-model.alpha))*sum(model.prob[s]*model.eta[s] for s in model.S)

    def con1(model, t, s):
        return model.P_DA[t] <= model.Pcap

    def con2(model, t, s):
        return model.Delta[t,s] == model.P_W_DA[t,s] - model.P_DA[t]

    def con3(model, t, s):
        return model.Delta[t,s] == model.Delta_plus[t,s] - model.Delta_minus[t,s]

    def con4(model, t, s):
        return model.Delta_plus[t,s] <= model.P_W_DA[t,s]

    def con5(model, t, s):
        return model.Delta_minus[t,s] <= model.Pcap

    def con_aux_1(model,t):
        return model.EIm_hourly[t] == sum(model.prob[s]*model.Delta[t,s] for s in model.S)


    #Add objective and constraints to the model
    model.objective = Objective(rule=obj, sense = maximize)
    model.con_1 = Constraint(model.T, model.S, rule= con1)
    model.con_2 = Constraint(model.T, model.S, rule= con2)
    model.con_3 = Constraint(model.T, model.S, rule= con3)
    model.con_4 = Constraint(model.T, model.S, rule= con4)
    model.con_5 = Constraint(model.T, model.S, rule= con5)

    model.con_6 = Constraint(model.S, rule = con_cvar1)
    model.con_7 = Constraint(rule= con_cvar2)
    model.con_8 = Constraint(rule= con_EP)

    model.con_9 = Constraint(model.S, rule=scenarioCost)
    model.con_aux_1 = Constraint(model.T, rule=con_aux_1)

    #Determine solver and solver options
    opt                    = SolverFactory('gurobi')
    opt.options['MIPgap']  = 0
    opt.options['threads'] = 0

    #Solve model
    results                = opt.solve(model)

    #Extract results
    mainResults = pandas.Series(index=['alpha', 'beta', 'expected_profit', 'CVaR', 'VaR'],
                                data=[model.alpha.value, model.beta.value, model.EP.value, model.cvar.value, model.zeta.value])

    bid = pandas.Series(index=model.T, data=[model.P_DA[t].value for t in model.T])
    bid.index.name = None

    imbalanceVolume = pandas.DataFrame(index = model.T, columns=model.S, data=None)
    for t in model.T:
        for s in model.S:
            imbalanceVolume.loc[t,s] = model.Delta[t,s].value

    resDist = pandas.DataFrame(index=model.S, columns=['profit', 'prob','cumprob'])
    for s in model.S:
        resDist.loc[s,'profit'] = model.scenario_cost[s].value
        resDist.loc[s, 'prob']  = model.prob[s]
        resDist.loc[s, 'cumprob'] = 0

    probs = resDist['prob']

    resDist.sort_values(by='profit', axis=0, ascending=True, inplace=True)
    for s_in, s in enumerate(resDist.index):
        if s_in > 0:
            resDist.loc[s, 'cumprob'] = resDist.loc[resDist.index[s_in-1], 'cumprob'] + resDist.loc[s, 'prob']
        else:
            resDist.loc[s, 'cumprob'] = resDist.loc[s, 'prob']

    hourlyExpectedImbalance = pandas.Series(index=model.T, data=[model.EIm_hourly[t].value for t in model.T])

    resList = [mainResults, bid, resDist, imbalanceVolume, hourlyExpectedImbalance, probs]

    return model, resList

