from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime
from django.shortcuts import render
from .models import Method
import os
import openpyxl
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import math
import random
import copy
from plotly.offline import plot
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

# Create your views here.

def home(request):
    
    return render(request, 'home.html')

def settings(request):
    Name = request.POST.get('Name')
    Selected = request.POST.getlist('method')
    Occur = [] #boolean list of method occurence
    if("移動平均 + 季節性" in Selected):
        Occur.append(True)
    else:
        Occur.append(False)
    if("一次指數平滑 + 季節性" in Selected):
        Occur.append(True)
    else:
        Occur.append(False)
    if("二次指數平滑 + 季節性" in Selected):
        Occur.append(True)
    else:
        Occur.append(False)
    if("移動平均(無季節性)" in Selected):
        Occur.append(True)
    else:
        Occur.append(False)
    if("一次指數平滑(無季節性)" in Selected):
        Occur.append(True)
    else:
        Occur.append(False)
    if("二次指數平滑(無季節性)" in Selected):
        Occur.append(True)
    else:
        Occur.append(False)
    if(Name != None):
        method, created = Method.objects.get_or_create(Name= Name)
        method.MA_S = Occur[0]
        method.SE_S = Occur[1]
        method.DE_S = Occur[2]
        method.MA_NS = Occur[3]
        method.SE_NS = Occur[4]
        method.DE_NS = Occur[5]
        method.save()



    return render(request, 'settings.html', {
        
    })

def imports(request):
    item  = Method.objects.all()
    Name = request.POST.get('Name')
    current_method = []
    if(Name != None):
        method_file= Method.objects.get(Name= Name)
        request.session['methods'] = [method_file.MA_S, method_file.SE_S, method_file.DE_S, method_file.MA_NS, method_file.SE_NS, method_file.DE_NS ]
        if(method_file.MA_S):
            current_method.append("移動平均 + 季節性")
        if(method_file.SE_S):
            current_method.append("一次指數平滑 + 季節性")
        if(method_file.DE_S):
            current_method.append("二次指數平滑 + 季節性")
        if(method_file.MA_NS):
            current_method.append("移動平均(無季節性)")
        if(method_file.SE_NS):
            current_method.append("一次指數平滑(無季節性)")
        if(method_file.DE_NS):
            current_method.append("二次指數平滑(無季節性)")
    else:
        Name = "未選擇方法"
    return render(request, 'imports.html', {'items': item,'current_method': current_method, 'method_name' : Name,
    })

def summary(request):
    Name = request.POST.get('Name')
    df_predict = pd.read_csv("templates/static/final.csv", index_col=0)
    df_origin = pd.read_csv("templates/static/original.csv", index_col=0)
    df_discard = pd.read_csv("templates/static/discarded.csv", index_col=0)
    discard_list = df_discard["商品編號"].tolist()
    discard_reason = df_discard["原因"].tolist()
    df_predict = df_predict.T
    if(Name != None and Name != '' and Name in df_predict.columns):
        
        NMAE = df_predict[Name].loc['nmae']
        plot_div = go.Figure()
        plot_div.add_scatter(x=[df_origin.index[-1], df_predict.index[0]], y=[df_origin[Name][-1], df_predict[Name][0]], mode='lines', line={'dash': 'dash', 'color': '#0000ff'}, showlegend = False, hoverinfo='none')
        plot_div.add_scatter(x=df_origin.index, y=df_origin[Name], mode='lines+markers', line_color='#0000ff', name="原始銷售")
        plot_div.add_scatter(x=df_predict.index[0:6], y=df_predict[Name][0:6], mode='lines+markers', line_color='#ff0000', name = "預測")
        plot_div.update_xaxes(title_text="月份")
        plot_div.update_yaxes(title_text="數量")
        plot_div.update_layout(
        title={
            'text': Name + " 預測結果 " + "NMAE: " + str(NMAE) + "<br>商品名稱: " + df_predict[Name].loc['商品名稱'],
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
        )
        plot_div.update_layout(xaxis_range=[df_origin.index[0],df_predict.index[5]])
        plot_div.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=6,
                     label="未來6個月",
                     step="month",
                     stepmode="backward"),
                dict(count=12,
                     label="6個月",
                     step="month",
                     stepmode="backward"),
                dict(count=18,
                     label="1年",
                     step="month",
                     stepmode="backward"),
                dict(count=42,
                     label="3年",
                     step="month",
                     stepmode="backward"),
                dict(count=66,
                     label="5年",
                     step="month",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
        output = plot(plot_div , output_type='div')
    else:
        if(Name in discard_list):
            
            plot_div = go.Figure()
            plot_div.update_layout(
            title={
                'text': Name + " 不預測原因:" + discard_reason[discard_list.index(Name)],
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
            )
            output = plot(plot_div , output_type='div')
        elif(Name == '' or Name == None):
            plot_div = go.Figure()
            plot_div.update_layout(
            title={
                'text': "請輸入貨號",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
            )
            output = plot(plot_div , output_type='div')

        else:
            plot_div = go.Figure()
            plot_div.update_layout(
            title={
                'text': "查無貨號",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
            )
            output = plot(plot_div , output_type='div')
    return render(request, "summary.html", context={'plot_div': output})

def importFile(request):
    message = "開始預測"
    if request.method =='POST':
        #若資料整行為 0, 則將 0 都改成 1
        #input : df -> dataframe
        #output : dataframe
        def preprocess(df):
            for i in df:
                if (df[i] == 0).any():
                    df[i] = 1

        #計算 mae
        #input : real -> series, pre -> series
        #output : scalar
        def mae_series(real, pre):
            real = real.reset_index(drop=True)
            pre = pre.reset_index(drop=True)
            return (real-pre).abs().mean()

        #計算 mape
        #input : real -> series, pre -> series
        #output : scalar
        def mape_series(real, pre):
            real = real.reset_index(drop=True)
            pre = pre.reset_index(drop=True)
            real = real.fillna(0)
            pre = pre.fillna(0)
            ins = real != 0
            real = real[ins]
            pre = pre[ins]
            
            return ((real-pre).abs() / real).mean()

        #計算 normalized mae
        #input : real -> series, pre -> series
        #output : scalar
        def nmae_series(real, pre):
            return mae_series(real, pre) / real.mean()

        #去除前幾期為 0 的資料，將該商品的銷售紀錄變成從開賣月份開始
        #input : siri -> series (某商品的原始資料)
        #output : series
        def shift_to_first_sells(siri):
            x = np.argmax(siri != 0)
            return siri[x:].reset_index(drop=True)

        #將資料 shift 指定期數
        #input : siri -> series, x -> scalar
        #output : series
        def shift(siri,x):
            return siri[x:].reset_index(drop=True)

        #計算該商品的 mae, mape 或 normalized mae
        #input : real -> series, pre -> series (預測值), shift_n -> scalar (shift 的期數 = n), loss -> string (欲計算的誤差指標 => ‘mae’, ‘mape’, ‘nmae’)
        #output : scalar (誤差值)
        def compute_series_loss(real, pre, shift_n=0, loss='mae'):
            if shift_n!= 0:
                real = shift(real, shift_n)

            if loss == 'mae':
                loss_func = mae_series(real, pre)
            elif loss == 'mape':
                loss_func = mape_series(real, pre)
            elif loss == 'nmae':
                loss_func = nmae_series(real, pre)

            return loss_func

        #根據給定的 n，計算該商品移動 n 期的預測值
        #input : siri -> series, n -> scalar
        #output : series (該商品每期的預測值)
        def predict_ma_series(siri, n=3, seasonal_idx=None):
            temp = []
            for i in range(n, siri.shape[0]):
                m = np.mean(siri[(i-n): (i)])
                temp.append(m)
            ma = pd.Series(temp)
            if type(seasonal_idx)!= type(None):
                seasonal_idx = seasonal_idx[n:].reset_index(drop=True)
                ma = ma * seasonal_idx
            
            return ma

        #計算該商品最佳的 n
        #input : siri -> series, lower_bound -> scalar (n 的最小值), upper_bound -> scalar (n的最大值)
        #output : scalar (最佳 n)
        def fit_ma_series(siri, lower_bound=1, upper_bound=6):
            best_n = "n"
            loss = "n"
            
            for i in range(lower_bound, upper_bound+1):
                pre = predict_ma_series(siri, i)
                if len(pre) != 0:
                    temp_loss = compute_series_loss(siri, pre, i, loss='mae')
                if len(pre) <= 6:
                    temp_loss = 1
                if(loss == "n" or temp_loss < loss):
                    loss = temp_loss
                    best_n = i 
                    
            return best_n

        #計算該商品經過 α 平滑後的預測值 (一次指數平滑)
        #input : siri -> series, alpha -> scalar
        #out : series (該商品每期的預測值)
        def predict_simple_xp_series(siri, alpha=0.5, seasonal_idx=None, alll=False):
            if(alll == False):
                pre_len = len(siri) - 1
            else:
                pre_len = len(siri)
            predict = pd.Series(range(pre_len), dtype=float, name="SXP")
            
            predict[0] = siri[0]
            for i in range(1, pre_len):
                prei = i-1
                predict[i] = alpha * siri[i] + (1-alpha) * predict[prei]
            if type(seasonal_idx) != type(None):
                seasonal_idx = seasonal_idx[1:].reset_index(drop=True)
                predict = predict * seasonal_idx
            
            return(predict)

        #計算該商品最佳的 α (一次指數平滑)
        #input : siri -> series, end -> scalar (α 最大值), start -> scalar (α 最小值), n -> scalar (最大值與最小值中的間隔數)
        #output : scalar (最佳 α)
        def fit_simple_xp_series(siri, end=0.9, start=0.1, n=9):
            best_alpha = "n"
            loss = "n"
            
            for i in np.linspace(start, end, n):
                
                pre = predict_simple_xp_series(siri, alpha=i)
                temp_loss = compute_series_loss(siri, pre, 1)

                if(best_alpha == "n" or temp_loss < loss):
                    loss = temp_loss
                    best_alpha = i
            
            return best_alpha

        #計算該商品經過 α 與 β 平滑後的預測值 (二次指數平滑)
        #input : siri -> series, alpha -> scalar, beta -> scalar
        #output : series (該商品每期的預測值)
        def predict_double_xp_series(siri, alpha=0.5, beta=0.5, seasonal_idx=None, alll=False):
            pre = predict_simple_xp_series(siri, alpha, alll=alll)
            pre2 = predict_simple_xp_series(pre, beta, alll=alll)
            if type(seasonal_idx) != type(None):
                seasonal_idx = seasonal_idx[2:].reset_index(drop=True)
                pre2 = pre2 * seasonal_idx
            
            return pre2

        #計算該商品最佳的 α 與 β (二次指數平滑)
        #input : siri -> series, end -> scalar (α、β 最大值), start -> scalar (α、β 最小值), n -> scalar (最大值與最小值中的間隔數)
        #output : tuple (α, β)
        def fit_double_xp_series(siri, end=0.9, start=0.1, n=9):
            best_alpha = "n"
            best_beta = "n"
            loss = "n"
            
            for i in np.linspace(start, end, n):
                pre = predict_simple_xp_series(siri, alpha=i)
                
                for j in np.linspace(start, end, n):
                    pre2 = predict_simple_xp_series(pre, alpha=j)
                    temp_loss = compute_series_loss(siri, pre2, 2)

                    if(best_alpha == "n" or temp_loss < loss):
                        loss = temp_loss
                        best_alpha = i
                        best_beta = j
                
            return (best_alpha,best_beta)

        #使用 MA 往下預測 n 期
        #input : siri -> series, p -> scalar (MA 算出來的最佳期數)
        #output : series (MA 的預測值)
        def ma_predict(siri, p, n = 5):
            siri = siri.reset_index(drop=True)
            y = pd.Series(np.zeros(len(siri)+n))
            for i in range(len(siri)):
                y[i] = siri[i]
            for i in range(len(siri), len(siri)+n):
                s = np.mean(y[i-p:i])
                y[i] = s
            return y

        #回傳 siri 每期對應的 seasonal index
        #input : siri -> series, length -> scalar
        #output : series
        def ret_full_sea(siri, length):
            y = pd.Series(np.zeros(length))
            for i in range(length):
                y[i] = siri[i%12]
            return y

        #找出該資料最佳的 n (MA), α (一次 ES), α 與 β (二次 ES)
        #input : siri -> series, start -> scalar (α 或 β 最小值), end -> scalar (α 或 β 最大值), ex_n -> scalar (最大值與最小值中的間隔數), ma_n_l -> scalar (n 的最小值), ma_n_u -> scalar (n 的最大值)
        #output : tuple (n, α, (α, β))
        def fit_all_series(siri, start=0.1, end=0.9, ex_n=9 , ma_n_l=1, ma_n_u=6):
            ma_param = fit_ma_series(siri, lower_bound=ma_n_l, upper_bound=ma_n_u)
            se_param = fit_simple_xp_series(siri, start, end, ex_n)
            de_param = fit_double_xp_series(siri, start, end, ex_n)
            return (ma_param, se_param, de_param)

        #計算出該商品在最佳 n, α 與 (α, β) 下的預測值
        #input : siri -> series, params -> tuple (n, α, (α, β))
        #output : tuple (series, series, series)
        def predict_all_series(siri, params, seasonal_idx=None):  #params = (ma_param, se_param, de_param)
            ma_param, se_param, de_param = params
            ma_siri = predict_ma_series(siri, ma_param, seasonal_idx)
            se_siri = predict_simple_xp_series(siri, se_param, seasonal_idx)
            de_siri = predict_double_xp_series(siri, de_param[0], de_param[1], seasonal_idx)
            return (ma_siri, se_siri, de_siri)

        #計算六種方法的 train nmae
        #input : siri -> series, real_sell -> series, param -> tuple (n, α, (α, β)), ns_param -> tuple (n, α, (α, β))
        #output : tuple (scalar, scalar, scalar, scalar, scalar, scalar)
        def return_train_nmae(siri, real_sell, param, ns_param, seasonal_idx=None):
            const = len(siri)
            train = real_sell[:const]
            
            # seasonal
            ma_pre = predict_ma_series(siri, param[0], seasonal_idx=seasonal_idx)
            ma_pre_train = ma_pre[0:const-param[0]]
            
            se_pre = predict_simple_xp_series(siri, param[1], seasonal_idx=seasonal_idx)
            se_pre_train = se_pre[0:const-1]
            
            de_pre = predict_double_xp_series(siri, param[2][0], param[2][1], seasonal_idx=seasonal_idx)
            de_pre_train = de_pre[0:const-2]
            
            ma_nmae = round(compute_series_loss(train, ma_pre_train, param[0], loss='nmae'),3)
            se_nmae = round(compute_series_loss(train, se_pre_train, 1, loss='nmae'),3)
            de_nmae = round(compute_series_loss(train, de_pre_train, 2, loss='nmae'),3)
            
            # not_seasonal -> ns
            ns_ma_pre = predict_ma_series(real_sell, ns_param[0])
            ns_ma_pre_train = ns_ma_pre[0:const-ns_param[0]]
            
            ns_se_pre = predict_simple_xp_series(real_sell, ns_param[1])
            ns_se_pre_train = ns_se_pre[0:const-1]
            
            ns_de_pre = predict_double_xp_series(real_sell, param[2][0], ns_param[2][1])
            ns_de_pre_train = ns_de_pre[0:const-2]
            
            ns_ma_nmae = round(compute_series_loss(train, ns_ma_pre_train, ns_param[0], loss='nmae'),3)
            ns_se_nmae = round(compute_series_loss(train, ns_se_pre_train, 1, loss='nmae'),3)
            ns_de_nmae = round(compute_series_loss(train, ns_de_pre_train, 2, loss='nmae'),3)
            
            return (ma_nmae, se_nmae, de_nmae, ns_ma_nmae, ns_se_nmae, ns_de_nmae)

        #使用 method_mask 中的預測方法往下預測 6 期, 並且產生 winner nmae
        #input : o1 -> series (原始資料), o2 -> series (seasonal index), o3 -> series (去季節性後的資料), method_mask -> array (1代表會使用該方法, 0代表不會, 順序為[MA_S, SE_S, DE_S, MA_NS, SE_NS, DE_NS])
        #output : (series (預測 6 期的值), series (winner nmae))
        def compute_six_month(o1, o2, o3, method_mask):
            ## seasonal
            s_new_o = shift_to_first_sells(o3)

            o_first_non_zero = len(o3) - len(s_new_o)
            n_first_test = int(len(s_new_o) * 0.75)

            param = fit_all_series(s_new_o[:n_first_test])

            ## no_seasonal

            ns_new_o = shift_to_first_sells(o1)

            o_first_non_zero = len(o1) - len(ns_new_o)
            n_first_test = int(len(ns_new_o) * 0.75)

            ns_param = fit_all_series(ns_new_o[:n_first_test])

            nmaes = return_train_nmae(s_new_o, ns_new_o, param, ns_param, seasonal_idx=o2)

            MA_S = 0
            SE_S = 1
            DE_S = 2
            MA_NS = 3
            SE_NS = 4
            DE_NS = 5

            mid = list(nmaes)
            mid = np.array(mid)
            mid = mid + 2000 * (1 - method_mask)
            mid = list(mid)
            min_method = np.argmin(np.array(mid))


            if min_method == MA_S:
                p = param[0]
                siri = s_new_o
                pre_before_sea = ma_predict(siri, p, 6)
                full_se_idx = ret_full_sea(o2, len(o1)+6)[o_first_non_zero:].reset_index(drop=True)
                ret = (pre_before_sea[-6:]*full_se_idx[-6:])

            if min_method == SE_S:
                p = param[1]
                siri = s_new_o
                pre = predict_simple_xp_series(siri, p, alll=True)
                pre_before_sea = ma_predict(pre, 3, 5)
                full_se_idx = ret_full_sea(o2, len(o1)+6)[1+o_first_non_zero:].reset_index(drop=True)

                ret = (pre_before_sea[-6:]*full_se_idx[-6:])

            if min_method == DE_S:
                p = param[2]
                siri = s_new_o
                pre = predict_double_xp_series(siri, p[0], p[1], alll=True)
                pre_before_sea = ma_predict(pre, 3, 4)
                full_se_idx = ret_full_sea(o2, len(o1)+6)[2+o_first_non_zero:].reset_index(drop=True)
                ret = (pre_before_sea[-6:]*full_se_idx[-6:])



            if min_method == MA_NS:
                p = ns_param[0]
                siri = ns_new_o
                pre = predict_ma_series(siri, p)
                ret = ma_predict(pre, p)[-6:].reset_index(drop=True)

            if min_method == SE_NS:
                p = ns_param[1]
                siri = ns_new_o
                pre = predict_simple_xp_series(siri, p)
                ret = ma_predict(pre, 3)[-6:].reset_index(drop=True)

            if min_method == DE_NS:
                p = ns_param[2]
                siri = ns_new_o
                pre = predict_double_xp_series(siri,p[0], p[1])
                ret = ma_predict(pre, 3)[-6:].reset_index(drop=True)

            return(ret, nmaes[min_method])

        #給定原始資料與seasonal index，計算往後的預測值
        #input : df -> series (原始資料), df2 -> series (seasonal index)
        #output : (dataframe [預測值(column 為貨號)], dataframe (winner nmae), dataframe [預測值(column 為日期)])
        def predict_df(df, df2):
            df = df.fillna(0)
            df = df.reset_index(drop=True)
            preprocess(df2)
            df4 = df2.copy()
            df3 = df.copy()
            ki = list(df.keys())
            while(df4.shape[0] < df.shape[0]):
                if(df4.shape[0] + 12 < df.shape[0]):
                    df4 = pd.concat((df4,df2.copy()), axis=0)
                else:
                    x = df.shape[0] - df4.shape[0]
                    df4 = pd.concat((df4,df2[:][:x].copy()), axis=0)
            df2 = df4
            df2 = df2.reset_index(drop=True)
            for k in ki:
                df3[k] = df[k] / df2[k]

            nki = ki.copy()
            discarded = []
            discarded_list = []
            for i in nki:
                if len(shift_to_first_sells(df[i])) < 6:
                    ki.remove(i)
                    discarded.append(i+",total sample < 6")
                    discarded_list.append((i, "total sample < 6"))
                    continue
                if df[i].sum() < 30:
                    ki.remove(i)
                    discarded.append(i+",total sells < 30")
                    discarded_list.append((i, "total sells < 30"))
                    continue
                if df[i][-4:].sum() == 0:
                    ki.remove(i)
                    discarded.append(i+",no sells between last 4 month")
                    discarded_list.append((i, "no sells between last 4 month"))
            # with open('discard.csv', 'w') as f:
            #     f.write('\n'.join(discarded))

            dd = pd.DataFrame()
            dd2 = pd.DataFrame()
            for idx, i in enumerate(ki[:]):
                method_array = np.array(request.session['methods'])
                o1 = df[i]
                o2 = df2[i]
                o3 = df3[i]
                try:
                    last_six, nmae = compute_six_month(o1,o2,o3,method_array)
                    last_six = last_six.reset_index(drop=True)
                    dd[i] = last_six
                    dd2[i] = [nmae]
                except:
                    discarded.append(i+",fails due to unknown reason")
                    discarded_list.append((i, "fails due to unknown reason"))

            dd = dd.apply(round)
            dd = dd.astype(int)
            return (dd, dd2, discarded_list)

        def ssss(df):
            data = df
            data.index=data["HDATE"]
            data = data.drop(data.columns[1], axis=1)
            data = data.T
            data = data.drop(data.index[0], axis=0)
            data.columns.name = ""
            data.insert(0, "HDATE", data.index)
            data.index = range(0, data.shape[0])
            month = data.shape[0]
            data = pd.DataFrame(data.drop(['HDATE', data.columns[0]],axis = 1).values, index = data['HDATE'].values, columns = data.drop(['HDATE', data.columns[0]],axis = 1).columns)
            data_fillna = data.fillna(0)
            Sum = []
            for i in range(data.shape[1]):
                a = np.sum(data_fillna.iloc[:,i])
                Sum.append(a)
            data_detail = data.append(pd.DataFrame([Sum],columns=data.columns))
            data_detail = data_detail.rename(index = {0 : 'sum'})
            data_detail = data_detail.sort_values(by = 'sum', axis = 1, ascending = 0)
            data_detail = data_detail.append(pd.DataFrame([range(data.shape[1])],columns=data_detail.columns))
            data_detail = data_detail.rename(index = {0 : 'sum_rank'})
            data_detail.loc['sum_rank'] += 1

            def seasonal_index(data, group, start, end):
                group_sum = np.zeros(12)
                group_count = np.zeros(12)
                group_index = np.zeros(12)
                seasonal_index = np.zeros(12).tolist()
                for i in range(start, end):
                    m = i%12
                    g = group[m]
                    group_count[g] = group_count[g] + 1
                    group_sum[g] = group_sum[g] + data[i]
                for i in range(12):
                    if group_count[i] == 0:
                        group_count[i] = 1
                    group_index[i] = group_sum[i] / group_count[i]
                for i in range(12):
                    seasonal_index[i] = group_index[group[i]]
                s = sum(seasonal_index)
                if s == 0:
                    s = 1
                for i in range(12):
                    seasonal_index[i] = round(seasonal_index[i] * 12 / s, 2)
                return seasonal_index

            def seasonal_index_score(data, index, start, end): #scored by r square
                I = copy.deepcopy(index)
                if(start == end):
                    return -100000, -100000
                D = np.zeros(month)
                for i in range(start, end):
                    m = i%12
                    if I[m] == 0:
                        I[m] = 1
                    D[i] = round(data[i] / I[m], 2)
                df = pd.DataFrame(range(end - start), columns = ['month'])
                df['data'] = D[start:end]
                result = smf.ols('data ~ month', data=df).fit()
                score = result.rsquared
                return score, result 

            def best_seasonal_index_topn(data, groups):
                amount = data.shape[1]
                Best = []
                for i in range(amount):
                    D = data[data.columns[i]][:month].fillna(0).tolist()
                    if sum(D) == 0:
                        Best.append([1,1,1,1,1,1,1,1,1,1,1,1])
                        continue
                    start = 0
                    best = [1,1,1,1,1,1,1,1,1,1,1,1]
                    best_score = -10000
                    for j in range(month):
                        if D[start] == 0 and start + 1 < month:
                            start = start + 1
                    end = month
                    for j in range(len(groups)):
                        group = groups[j]  
                        seasonal = seasonal_index(D, group, start, end)
                        score, result = seasonal_index_score(D, seasonal, start, end)
                        if best_score < score:
                            best = seasonal
                            best_score = score
                    Best.append(best)
                return Best

            groups = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11]
                    ,[0, 1, 2, 3, 3, 5, 6, 7, 8, 9, 9,11] #母親節，周年慶高峰
                    ,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #無
                    ]
            groups_detail = ['無'
                            ,'母親節，周年慶高峰'
                            ,'剩'
                            ]
            I = [1,2,3,4,5,6,7,8,9,10,11,12]
            sum_index = best_seasonal_index_topn(data_detail, groups)
            Sum_index = pd.DataFrame(sum_index, index = data_detail.columns, columns = I).transpose()
            return Sum_index

        def preprocessing(data):
            clim_s = list(data.columns).index("CLIM")
            vitel_s = list(data.columns).index("VITEL")
            num = vitel_s - clim_s
            data.columns = data[0:1].values.tolist()[0]
            data = data.drop(0)
            data = data.reset_index(drop=True)
            
            #算比例 => ratio
            sales = data.iloc[:, range(clim_s, vitel_s)].fillna(0)
            v = data.iloc[:, range(vitel_s, vitel_s+num)].fillna(0)
            ratio = pd.DataFrame(np.zeros((sales.shape[0], 3)), columns=["商品貨號", "CLIM", "VITEL"])
            c_cnt = sales.sum(axis=1).array
            v_cnt = v.sum(axis=1).array
            for i in range(sales.shape[0]):
                if v_cnt[i] == 0:
                    ratio["CLIM"][i] = 1
                    ratio["VITEL"][i] = 0
                else:
                    ratio["CLIM"][i] = c_cnt[i] / v_cnt[i]
                    ratio["VITEL"][i] = 1
            ratio["商品貨號"] = data["商品貨號"]
            ratio["CLIM/ALL"] = ratio["CLIM"] / (ratio["CLIM"] + ratio["VITEL"])
            ratio = ratio.set_index('商品貨號')
            
            #合併銷售資料 => sales
            sales = sales + v
            sales.insert(0, "HDATE", data["商品貨號"])
            
            category = data.iloc[:, [0, 1, 2, 3]]
            category = category.set_index("商品貨號")
            
            return ratio, sales, category
        
        def past_sales(sales):
            #mean_3
            all3 = sales.iloc[:, range(sales.shape[1]-3, sales.shape[1])]
            mean_3 = pd.Series(all3.mean(axis=1), name = "past_3")
            mean_3 = mean_3.to_frame()
            mean_3.insert(0, "product", sales["HDATE"])
            
            #mean_12
            all12 = sales.iloc[:, range(sales.shape[1]-12, sales.shape[1])]
            mean_12 = pd.Series(all12.mean(axis=1), name = "past_12")
            mean_12 = mean_12.to_frame()
            mean_12.insert(0, "product", sales["HDATE"])
            
            #same_period
            month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            index = sales.columns[-1].find("-")
            m = month.index(sales.columns[-1][:index])
            y = sales.columns[-1][index+1:]
            
            m_list = []
            if m + 6 < 12:
                for i in range(m+1, m+7):
                    m_list.append(str(month[i]) + "-" + str(int(y)-1))
            else:
                t = 11 - m
                for i in range(m+1, 12):
                    m_list.append(str(month[i]) + "-" + str(int(y)-1))
                for i in range(0, 6-t):
                    m_list.append(str(month[i]) + "-" + str(y))
            
            start = list(sales.columns).index(m_list[0])
            end = list(sales.columns).index(m_list[-1])
            alls = sales.iloc[:, range(start, end+1)]
            same_period = pd.Series(alls.mean(axis=1), name = "same_period")
            same_period = same_period.to_frame()
            same_period.insert(0, "product", sales["HDATE"])
            
            mean_3 = mean_3.set_index('product').T
            mean_12 = mean_12.set_index('product').T
            same_period = same_period.set_index('product').T
            
            agg = mean_3.append([mean_12, same_period])
            agg.index = (["過去三個月平均銷售", "過去十二個月平均銷售", "去年同期平均銷售"])
            return agg

        def final_predict(input_df):
            data = input_df.copy()
            ratio, sales, category = preprocessing(data)
            seasonal_index = ssss(sales)
            
            df = sales.transpose()
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            
            # change df.index into datetime to create predict index
            new_date = []
            for i in range(len(df.index)):
                new_date.append(datetime.strptime(df.index[i], "%b-%y"))
            
            end_month = new_date[-1]
            predict_start = end_month + relativedelta(months=1) # next month of original data
            predict_index = pd.date_range(start=predict_start, periods=6, freq='MS')
            predict_index = np.datetime_as_string(predict_index, unit='D')
            #print(predict_index)
            
            predict_sales, nmae, discarded_list = predict_df(df, seasonal_index)
            df['HDATE'] = new_date
            df.set_index('HDATE', drop = True, inplace = True)
            df.to_csv('templates/static/original.csv')
            ratio_dic = {}
            for i in ratio.T:
                ratio_dic[i] = ratio.T[i][2]
            category = category.T
            
            clim = predict_sales.copy()
            for i in clim:
                clim[i] = round(clim[i] * ratio_dic[i])
            vitel = predict_sales.copy() - clim
            
            predict_sales.index = predict_index
            clim.index = ["CLIM_" + x for x in predict_index]
            vitel.index = ["VITEL_" + x for x in predict_index]
            nmae.index = (["nmae"])

            # add average figure agg
            agg = past_sales(sales.copy())
            
            final_df = predict_sales.append([clim, vitel, nmae, category[list(predict_sales.keys())], agg[list(predict_sales.keys())]]).T
            
            if(len(discarded_list)>0):
                ki, reason = list(zip(*discarded_list))
                
                discarded_df = pd.DataFrame({
                    "商品編號": ki,
                    "原因": reason
                })
            else:
                discarded_df = pd.DataFrame({
                    "商品編號": [],
                    "原因": []
                })
            
            return final_df, seasonal_index, discarded_df
        data = pd.read_excel(request.FILES['file_upload'])
        final_df, seasonal_index, discarded = final_predict(data)
        message = "預測完成"
        final_df.to_csv("templates/static/final.csv", index = True, header = True)
        discarded.to_csv("templates/static/discarded.csv", index = True, header = True)
        
    return render(request, 'importFile.html', {'message': message
        
    })

