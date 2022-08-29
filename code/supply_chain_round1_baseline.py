import time
import pandas as pd
import numpy as np
import datetime
from pandas.tseries.offsets import DateOffset
import copy
import gurobipy as gp


def maxSubArray(nums):
    """ :type nums: List[int] :rtype: int """
    if len(nums) < 2:
        return nums[0]
    dp = [nums[0] for i in range(len(nums))]
    max_result = nums[0]  # 最开始的是nums[0]，后面如果是负数肯定更小，如果是整数肯定变大
    for i in range(1, len(nums)):
        if dp[i-1] < 0:
            dp[i] = nums[i]
        else:
            dp[i] = dp[i-1] + nums[i]
        if max_result < dp[i]:
            max_result = dp[i]
    return max_result

class ReplenishUnit:
    def __init__(self,
                 unit,
                 demand_hist,
                 intransit,
                 qty_replenish,
                 qty_inventory_today,
                 qty_using_today,
                 arrival_sum,
                 lead_time
                 ):
        '''
        记录各补货单元状态
        :param unit:
        :param demand_hist: 净需求历史
        :param intransit: 补货在途
        :param qty_replenish: 补货记录
        :param qty_inventory_today: 当前可用库存
        :param qty_using_today: 当前已用库存（使用量）
        :param arrival_sum: 补货累计到达
        :param lead_time: 补货时长，交货时间
        '''
        self.unit = unit
        self.demand_hist = demand_hist
        self.intransit = intransit
        self.qty_replenish = qty_replenish
        self.qty_inventory_today = qty_inventory_today
        self.qty_using_today = qty_using_today
        self.arrival_sum = arrival_sum
        # self.qty_using_today
        self.lead_time = lead_time
        self.demand_pred_range = []

    def update(self,
               date,
               arrival_today,
               demand_today,
               demand_pred_range):
        '''
        每日根据当天补货到达与当日净需求更新状态
        :param date:
        :param arrival_today: 当天补货到达
        :param demand_today: 当天净需求
        :return:
        '''
        self.demand_pred_range = demand_pred_range

        self.qty_inventory_today += arrival_today
        self.arrival_sum += arrival_today
        inv_today = self.qty_inventory_today
        if demand_today < 0:
            self.qty_inventory_today = self.qty_inventory_today + min(-demand_today, self.qty_using_today)
        else:
            self.qty_inventory_today = max(self.qty_inventory_today - demand_today, 0.0)
        self.qty_using_today = max(self.qty_using_today + min(demand_today, inv_today), 0.0)
        self.demand_hist = self.demand_hist.append({"ts": date, "unit": self.unit, "qty": demand_today}, ignore_index = True)


    def forecast_function(self,
                          demand_hist):
        demand_average = np.mean(self.demand_hist["qty"].values[-3 * self.lead_time:])
        return [demand_average] * 90

    def replenish_function(self,
                           date):
        '''
        根据当前状态判断需要多少的补货量
        补货的策略由选手决定，这里只给一个思路
        :param date:
        :return:
        '''
        replenish = 0.0
        if date.dayofweek != 0:
            #周一为补货决策日，非周一不做决策
            pass
        else:
            #预测未来需求量
            # qty_demand_forecast = self.forecast_function(demand_hist = self.demand_hist)
            qty_demand_forecast = self.demand_pred_range
            #计算在途的补货量
            qty_intransit = sum(self.intransit) - self.arrival_sum
            r = self.get_opt_replenish(qty_demand_forecast, qty_intransit,date)
            replenish, obj_opt = r[0], r[1]

            # #安全库存 用来抵御需求的波动性 选手可以换成自己的策略
            # safety_stock = (max(self.demand_hist["qty"].values[-3 * self.lead_time:]) - (np.mean(self.demand_hist["qty"].values[- 3 * self.lead_time:]))) * self.lead_time
            demand_standard_deviation = np.std(self.demand_hist['qty'].values[-3 * self.lead_time:])



            #
            # #再补货点，用来判断是否需要补货 选手可以换成自己的策略
            if not len(qty_demand_forecast):
                self.qty_replenish.at[date] = 0
                self.intransit.at[date + self.lead_time * date.freq] = 0
            else:
                # policy 1
                # safety_stock = 0.5 * demand_standard_deviation * np.sqrt(self.lead_time)
                # self.qty_replenish.at[date] = replenish + safety_stock
                # self.intransit.at[date + self.lead_time * date.freq] = replenish +safety_stock
                # policy 2
                # safety_stock2 = 1* demand_standard_deviation
                # self.qty_replenish.at[date] = replenish + safety_stock2*np.sqrt((len(qty_demand_forecast)-self.lead_time+1))
                # self.intransit.at[date + self.lead_time * date.freq] = replenish + safety_stock2*np.sqrt((len(qty_demand_forecast)-self.lead_time+1))
                # p3
                safety_stock3 = 1.04 * demand_standard_deviation
                reorder_point = sum(qty_demand_forecast[(self.lead_time-1):])
                ss = safety_stock3*np.sqrt((len(qty_demand_forecast)-self.lead_time+1))
                replenish = (replenish+(reorder_point+ss))/2
                print('replenish is -------------------     ', replenish, '---------------------renew replenish')
                # 判断是否需要补货并计算补货量，选手可以换成自己的策略，可以参考赛题给的相关链接
                # if self.qty_inventory_today + qty_intransit < reorder_point:
                #     replenish = reorder_point - (self.qty_inventory_today + qty_intransit)
                self.qty_replenish.at[date] = replenish
                self.intransit.at[date + self.lead_time * date.freq] = replenish

    def get_opt_replenish(self,
                          qty_demand_forecast,
                          qty_intransit,
                          date):
        if not len(qty_demand_forecast):
            return 0, 0
        else:
            # update using demand_pred
            d_list_1 = pd.date_range(start=date+pd.DateOffset(days=1), end=date+pd.DateOffset(days=self.lead_time-1))
            d_list_2 = pd.date_range(start=date+pd.DateOffset(days=self.lead_time), end=date+pd.DateOffset(days=len(qty_demand_forecast)))

            qty_inventory_today = copy.deepcopy(self.qty_inventory_today)
            qty_using_today = copy.deepcopy(self.qty_using_today)
            for ind, d in enumerate(d_list_1):
                qty_inventory_today += self.intransit.get(d, default=0)
                inv_today = qty_inventory_today
                demand_today = qty_demand_forecast[ind]
                if demand_today < 0:
                    qty_inventory_today = qty_inventory_today + min(-demand_today, qty_using_today)
                else:
                    qty_inventory_today = max(qty_inventory_today - demand_today, 0)
                qty_using_today = max(qty_using_today + min(demand_today, inv_today), 0)


            # opt at replenish day
            demand_sequence = np.array(qty_demand_forecast[self.lead_time-1:])
            mss = maxSubArray(demand_sequence)
            big_num = gp.GRB.INFINITY # 1e10

            model = gp.Model('replenish')
            var_replenish = model.addVar(lb=0) # 补货日补货量
            var_qty_inventory = model.addVars(len(demand_sequence)) # 到货日开始到预测结束日的库存量
            var_qty_using = model.addVars(len(demand_sequence)) # 到货日开始到预测结束日的使用量
            var_stockout = model.addVars(len(demand_sequence)) # 到货日开始到预测结束日的未满足量
            qty_us_td = model.addVar()

            model.update()
            model.addConstr(qty_us_td == qty_using_today)
            for ind, d in enumerate(d_list_2):
                demand_today = demand_sequence[ind]
                dt = model.addVar(lb=-gp.GRB.INFINITY)
                model.addConstr(dt == demand_today)
                if ind == 0: # 14天后的到货日
                    if demand_today < 0:
                        z1 = model.addVar()
                        model.addConstr(z1 == -1 * demand_today)
                        z2 = model.addVar(ub=big_num)
                        model.addConstr(z2 == gp.min_(z1, qty_us_td)) # qty_using_today是昨天的用量，z2是今天的释放量
                        model.addConstr(var_qty_inventory[ind] == var_replenish + qty_inventory_today + z2) # qty_inventory_today是昨天的库存
                    else:
                        k1 = model.addVar(lb=-big_num, ub=big_num)
                        model.addConstr(k1 == qty_inventory_today + var_replenish - demand_today)
                        model.addConstr(var_qty_inventory[ind] == gp.max_(k1, 0))
                    k2 = model.addVar()
                    model.addConstr(k2 == qty_inventory_today + var_replenish)
                    k3 = model.addVar(lb=-big_num, ub=big_num)
                    model.addConstr(k3 == gp.min_(k2,dt))
                    k4 = model.addVar(lb=-big_num, ub=big_num)
                    model.addConstr(k4 == qty_using_today + k3)
                    model.addConstr(var_qty_using[ind] == gp.max_(k4, 0))
                    k5 = model.addVar(lb=-big_num, ub=big_num)
                    model.addConstr(k5 == demand_today - (qty_inventory_today + var_replenish))
                    model.addConstr(var_stockout[ind] == gp.max_(k5, 0))
                # 非到货日
                else:
                    if demand_today < 0:
                        y1 = model.addVar()
                        model.addConstr(y1 == -1 * demand_today)
                        y2 = model.addVar()
                        model.addConstr(y2 == gp.min_(y1, var_qty_using[ind - 1]))
                        model.addConstr(var_qty_inventory[ind] == var_qty_inventory[ind - 1] + y2)
                    else:
                        x0 = model.addVar(lb=-big_num, ub=big_num)
                        model.addConstr(x0 == var_qty_inventory[ind - 1] - demand_today)
                        model.addConstr(var_qty_inventory[ind] == gp.max_(x0, 0))

                    x2 = model.addVar(lb=-big_num, ub=big_num)
                    model.addConstr(x2 == gp.min_(dt, var_qty_inventory[ind - 1]))
                    x3 = model.addVar(lb=-big_num, ub=big_num)
                    model.addConstr(x3 == x2 + var_qty_using[ind - 1])
                    model.addConstr(var_qty_using[ind] == gp.max_(x3, 0))
                    x4 = model.addVar(lb=-big_num, ub=big_num)
                    model.addConstr(x4 == demand_today - var_qty_inventory[ind - 1])
                    model.addConstr(var_stockout[ind] == gp.max_(x4, 0))

            sla = model.addVar(lb=-big_num)
            if mss < 1e-6:
                print('?????????????????? ---   mss = 0')

            model.addConstr(mss * (1 - sla) == gp.quicksum(var_stockout))
            o11 = model.addVar(lb=-5, ub=5)
            model.addConstr(o11 == -10 * (sla - 0.5))
            o12 = model.addVar(ub=150)
            model.addGenConstrExp(o11, o12)
            f_sla = model.addVar(lb=0, ub=1)
            model.addConstr(f_sla * (1 + o12) == 1)
            inv_rate = model.addVar(lb=0, ub=1)
            inv_rate_for_each_day = model.addVars(len(demand_sequence), ub=1, lb=0)
            model.addConstrs(inv_rate_for_each_day[i] * (var_qty_inventory[i] + var_qty_using[i]) == var_qty_inventory[i] for i in range(len(demand_sequence)))
            T_ = 1 / len(demand_sequence)
            model.addConstr(inv_rate == T_ * gp.quicksum(inv_rate_for_each_day))
            model.setObjective(0.5 * f_sla + 0.5 * (1 - inv_rate), gp.GRB.MAXIMIZE)

            # model.Params.LogToConsole = True  # 显示求解过程
            # MODEL.Params.MIPGap = 0.0001  # 百分比界差
            model.Params.TimeLimit = 50  # 限制求解时间为 100s
            model.setParam('NonConvex', 2)
            model.setParam('OutputFlag', 0)

            # start_time = time.time()
            model.optimize()
            # end_time = time.time()
            # print('============================================')
            # print('unit num. ' + str(ind) + ' ' + str(self.unit))
            # print('date is +++++++++  '+str(date)+'+++++++++++' )
            # print('opt time is ', str(end_time - start_time))


            try:
                # print('var_replenish.x  ', var_replenish.x, '**********************' )
                # print('model obj is ', model.getObjective().getValue())
                # print('============================================')
                return var_replenish.x, model.getObjective().getValue()
            except:
                with open('failure_opt.txt', 'a+') as f:
                    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
                    f.write(str(now) + ' ' + self.unit + ' ' + str(date) + '\n')
                    f.close()
                print('~~~~~~~~~~~~===============~~~~~~~~~~~~~~')
                print('fail to opt with unit of ', str(self.unit))
                print('~~~~~~~~~~~~===============~~~~~~~~~~~~~~')
                return mss, 0



class SupplyChainRound1Baseline:
    def __init__(self):
        self.using_hist = pd.read_csv("../data/Dataset/demand_train_B.csv")
        self.using_future = pd.read_csv("../data/Dataset/demand_test_B.csv")
        self.inventory = pd.read_csv("../data/Dataset/inventory_info_B.csv")
        self.last_dt = pd.to_datetime("20210301")
        self.start_dt = pd.to_datetime("20210302")
        self.end_dt = pd.to_datetime("20210607")
        self.lead_time = 14
        # self.using_pred = pd.read_csv("ss_1.csv")
        self.using_pred = pd.read_csv("ss_lgb_cat1605.csv")

    def run(self):
        self.using_hist["ts"] = self.using_hist["ts"].apply(lambda x:pd.to_datetime(x))
        self.using_future["ts"] = self.using_future["ts"].apply(lambda x:pd.to_datetime(x))
        self.using_pred['ts'] = self.using_pred['ts'].apply(lambda x:pd.to_datetime(x))
        qty_using = pd.concat([self.using_hist, self.using_future])
        date_list = pd.date_range(start = self.start_dt, end = self.end_dt)
        unit_list = self.using_future["unit"].unique()
        res = pd.DataFrame(columns = ["unit", "ts", "qty"])

        replenishUnit_dict = {}
        demand_dict = {}
        demand_dict_pred = {}

        #初始化，记录各补货单元在评估开始前的状态
        for chunk in qty_using.groupby("unit"):
            unit = chunk[0]
            demand = chunk[1]
            demand.sort_values("ts", inplace = True, ascending = True)

            #计算净需求量
            demand["diff"] = demand["qty"].diff().values
            demand["qty"] = demand["diff"]
            del demand["diff"]
            demand = demand[1:]
            replenishUnit_dict[unit] = ReplenishUnit(unit = unit,
                                                     demand_hist = demand.loc[demand["ts"] < self.start_dt],
                                                     intransit = pd.Series(index = date_list.tolist(), data = [0.0] * (len(date_list))),
                                                     qty_replenish = pd.Series(index = date_list.tolist(), data = [0.0] * (len(date_list))),
                                                     qty_inventory_today = self.inventory.loc[self.inventory["unit"] == unit]["qty"].values[0],
                                                     qty_using_today = self.using_hist.loc[(self.using_hist["ts"] == self.last_dt) & (self.using_hist["unit"] == unit)]["qty"].values[0],
                                                     arrival_sum = 0.0,
                                                     lead_time = self.lead_time)

            #记录评估周期内的净需求量
            demand_dict[unit] = demand.loc[(demand["unit"] == unit) & (demand["ts"] >= self.start_dt)].copy()
            demand_dict_pred[unit] = self.using_pred.loc[(self.using_pred["unit"] == unit) & (self.using_pred["ts"] >= self.start_dt)].copy().sort_values('ts', inplace=False, ascending=True)

        for date in date_list:
            #按每日净需求与每日补货到达更新状态，并判断补货量
            for unit in unit_list:
                # if unit!='3df22f447370785624a71797bf2b70f1':
                #     pass
                # else:
                #     if date ==  pd.to_datetime('2021-05-24'):
                #         print(1)
                # 真实的未来demand
                demand = demand_dict[unit]
                demand_today = demand[demand["ts"] == date]["qty"].values[0]
                demand_pred = demand_dict_pred[unit]
                if date+pd.DateOffset(days=20)<=date_list[-1]:
                    demand_pred_range = demand_pred.loc[(demand_pred['ts']>date) & (demand_pred['ts']<=date+pd.DateOffset(days=20))]['qty'].values
                elif date+pd.DateOffset(days=14) <= date_list[-1]:
                    demand_pred_range = demand_pred.loc[(demand_pred['ts']>date) & (demand_pred['ts']<=date_list[-1])]['qty'].values
                else:
                    demand_pred_range = []
                arrival = replenishUnit_dict[unit].intransit.get(date, default = 0.0)
                replenishUnit_dict[unit].update(date = date,
                                                arrival_today = arrival,
                                                demand_today = demand_today,
                                                demand_pred_range = demand_pred_range)
                replenishUnit_dict[unit].replenish_function(date)

        for unit in unit_list:
            res_unit = replenishUnit_dict[unit].qty_replenish
            res_unit = pd.DataFrame({"unit": unit,
                                     "ts": res_unit.index,
                                     "qty": res_unit.values})
            res_unit = res_unit[res_unit["ts"].apply(lambda x:x.dayofweek == 0)]
            res = pd.concat([res, res_unit])
        #输出结果
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        # res.to_csv("timely_baseline_" + now + ".csv")
        res.loc[res['qty'] < 0, 'qty'] = 0
        res.to_csv(("../submit/submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), header=None,
                    index=False)
        #res.to_csv("submit_timely_baseline_" + now + "_2.csv", index=False)

if __name__ == '__main__':
    supplyChainRound1Baseline = SupplyChainRound1Baseline()
    supplyChainRound1Baseline.run()
