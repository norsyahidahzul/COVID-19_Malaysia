#!/usr/bin/env python
# coding: utf-8

# # Simulation & Parameter Estimation of SEIRD Model
# ### Historical data: 27/2/2020 - 23/2/2021 (363 days)
# #### Phase1: 27/2/20 - 17/3/20 (Before MCO)
# #### Phase2: 18/3/20 - 9/6/20  (MCO+CMCO: 4/5/20-9/6/20)
# #### Phase3: 10/6/20 - 23/2/2021 (RMCO/consider as lockdown lifted, more economical and social activities are allowed, more flexibility)

# #### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ## Part 1: Simulation of Classical SEIRD Model (non-optimised parameters)
# ### Import module/library

# In[1]:


import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from tabulate import tabulate
import plotly.graph_objects as go
import plotly.io as pio
from lmfit import minimize, Parameters, report_fit
pio.renderers.default = "notebook"
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# ### Import Data from Excel

# In[2]:


#Import data from day 0 (before implementation of MCO) until 23/2/2021-day before immunisation programme starts)
df = pd.read_excel (r'C:\Users\USER\OneDrive - International Islamic University Malaysia\Desktop\SyahidZul\Master2021\Data Collection\Data_Covid19_MalaysiaGeneral.xlsx') 
covid_history = df


# In[3]:


covid_history.iloc[0:110]


# ### Historical Data Plot Malaysian COVID-19 cases with shaded region of each restricted movement phases

# In[4]:


fig = go.Figure()


fig.add_trace(go.Scatter(x=df['date'], y=covid_history.iloc[0:363].total_confirmed_cases, mode='markers+lines', marker_color='chocolate',        name='Total confirmed cases'))
fig.add_trace(go.Scatter(x=df['date'], y=covid_history.iloc[0:363].total_recovered_cases, marker_color='orange', mode='markers+lines',        name='Total recovered cases'))
fig.add_trace(go.Scatter(x=df['date'], y=covid_history.iloc[0:363].total_death_cases, mode='markers+lines', marker_color='red',        name='Total deaths cases'))
fig.add_trace(go.Scatter(x=df['date'], y=covid_history.iloc[0:363].daily_active_cases, mode='markers+lines', marker_color='purple',        name='Daily active cases'))


#DataFrame. iloc. Purely integer-location based indexing for selection by position. . iloc[]
#is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.

# Edit the layout
fig.update_layout(title='<b>COVID-19 Cases in Malaysia</b>', titlefont_family="Times News Roman",
                       xaxis_title='Days',
                       yaxis_title='Populations',
                       title_x=0.5, font_size=26,
                       width=1500, height=800, xaxis_range=[df['date'][0],df['date'][362]], yaxis_range=[0,300000]
                 )



fig.update_xaxes(tickangle=0, tickformat = '%d<Br> %B <Br>%Y' , tickmode='array')


#Add on shading for each phases (Pre-MCO, MCO, CMCO,RMCO)/shape regions
fig.add_vrect(
    x0=df['date'][0], x1=df['date'][19], annotation_text='<b>Pre-MCO</b>', annotation_textangle=90,annotation_bgcolor='lightsalmon', annotation_position="top left",
    fillcolor="LightSalmon", opacity=0.30, layer="below", line_width=0,
    ),

fig.add_vrect(
    x0=df['date'][20], x1=df['date'][67], annotation_text='<b>MCO</b>', annotation_textangle=90, annotation_bgcolor='cyan', annotation_position="top left",
    fillcolor="cyan", opacity=0.30, layer="below", line_width=0,
    ),

fig.add_vrect(
    x0=df['date'][68], x1=df['date'][103], annotation_text='<b>CMCO</b>', annotation_textangle=90, annotation_bgcolor='khaki', annotation_position="top left",
    fillcolor="khaki", opacity=0.50, layer="below", line_width=0,
    ),

fig.add_vrect(
    x0=df['date'][104], x1=df['date'][362], annotation_text='<b>RMCO</b>', annotation_textangle=90, annotation_bgcolor='lawngreen', annotation_position="top left",
    fillcolor="lawngreen", opacity=0.30, layer="below", line_width=0,
    )

fig.write_image("images/historicaldata_eachphases_general.png")
fig.show()


# ### Classical SEIRD model simulation (non-optimised constant params)

# In[8]:


#This is the classical SEIRD model
#no sporadic cases have been considered (only use one infection rate beta)
#no reinfection cases (no reinfection rate delta)
#constant parameters used (beta, sigma, gamma, mu)

#define Model
def ode_model(z, t, beta, sigma, gamma, mu):
    """
    Reference https://www.idmod.org/docs/hiv/model-seir.html
    """
    S, E, I, R, D = z
    N = S + E + I + R + D
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I - mu*I
    dRdt = gamma*I
    dDdt = mu*I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

#define ODE Solver
def ode_solver(t, initial_conditions, params):
    initS, initE, initI, initR, initD = initial_conditions
    beta, sigma, gamma, mu= params['beta'].value, params['sigma'].value, params['gamma'].value, params['mu'].value
    res = odeint(ode_model, [initS, initE, initI, initR, initD], t, args=(beta, sigma, gamma, mu)) 
    return res                                   #args used to pass a variable number of arguments to a function



#initial condition and initial values of parameters
#initN (Malaysian Population 2020- include non citizen)
initN = 32657300
initE = 3375  #ParticipantTablighwhoPositive/totalscreeningat27/2/20
initI = 1
initR = 22
initD = 0
initS = initN - (initE + initI + initR + initD)
beta= 12.5#15.44 
sigma = 0.19
gamma = 0.279 
mu = 0.1

days = 101

params = Parameters()
params.add('beta', value=beta, min=0, max=100)
params.add('sigma', value=sigma, min=0, max=100)
params.add('gamma', value=gamma, min=0, max=100)
params.add('mu', value=mu, min=0, max=100)

initial_conditions = [initS,initE, initI, initR, initD]
params['beta'].value, params['sigma'].value,params['gamma'].value, params['mu'].value = [beta, sigma, gamma, mu]
tspan = np.arange(0, days, 1) #timespan,np.arrange to arrange day 0 till days with increment of 1
sol = ode_solver(tspan, initial_conditions, params)
S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]


fig = go.Figure()

fig.add_trace(go.Scatter(x=tspan, y=S, mode='lines',line_color='blue', name='Susceptible'))
fig.add_trace(go.Scatter(x=tspan, y=E, mode='lines',line_color='turquoise', name='Exposed'))
fig.add_trace(go.Scatter(x=tspan, y=I, mode='lines', line_color='purple', name='Infected'))
fig.add_trace(go.Scatter(x=tspan, y=R, mode='lines', line_color='orange',name='Recovered'))
fig.add_trace(go.Scatter(x= tspan, y=D, mode='lines', line_color='red',name='Death'))
    

if days <= 30:
    step = 1
elif days <= 90:
    step = 7 
else:
    step = 10


# Edit the layout
fig.update_layout(title='Simulation of Classical SEIRD Model',
            xaxis_title='Days',
            yaxis_title='Populations',
            title_x=0.5, font_size= 22,
            width=1000, height=600
                     )
fig.update_xaxes(tickangle=0, tickformat = None ,tickmode='array', tickvals=np.arange(0, days + 1,step))


fig.write_image("images/Cseird_simulation_withoutHD_non-optimised_constantparams.png")
fig.show()

#suggestion: better to change days to date similar like plotting figures for historical data


# ### Classical SEIRD model simulation includes historical data (actual and forecasted data) -14 days

# In[7]:


t=days
days=14

initial_conditions = [initS,initE, initI, initR, initD]
params['beta'].value, params['sigma'].value,params['gamma'].value, params['mu'].value,  = [beta, sigma, gamma, mu]
tspan = np.arange(0, days, 1) #timespan,np.arrange to arrange day 0 till days with increment of 1
sol = ode_solver(tspan, initial_conditions, params)
S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]

fig = go.Figure()
#plotting simulated data
#fig.add_trace(go.Scatter(x=tspan, y=S, mode='lines',line_color='blue', name='Susceptible'))
#fig.add_trace(go.Scatter(x=tspan, y=E, mode='lines',line_color='turquoise', name='Exposed'))
fig.add_trace(go.Scatter(x=tspan, y=I, mode='lines', line_color='purple', name='Simulated Infected'))
fig.add_trace(go.Scatter(x=tspan, y=R, mode='lines', line_color='orange',name='Simulated Recovered'))
fig.add_trace(go.Scatter(x= tspan, y=D, mode='lines', line_color='red',name='Simulated Death'))
    
#plotting observed data #use 0 till 14 just to allow tick 14 to appear during simulation only. 
                              #during fitting will be using [0:13]
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].daily_active_cases, mode='markers',marker_color='purple',                             name='Observed Infected Cases', line = dict(dash='dash')))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].total_recovered_cases, mode='markers',marker_color='orange',                             name='Observed Recovered Cases', line = dict(dash='dash')))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].total_death_cases, mode='markers', marker_color='red',                             name='Observed Deaths Cases', line = dict(dash='dash')))

if days <= 30:
    step = 1
elif days <= 90:
    step = 7 
else:
    step = 40


# Edit the layout
fig.update_layout(title='Simulation of Classical SEIRD Model',
            xaxis_title='Days',
            yaxis_title='Populations',
            title_x=0.5, font_size= 26,
            width=1200, height=600
                     )
fig.update_xaxes(tickangle=0, tickformat = None, tickmode='array', tickvals=np.arange(0, days+1,step))

fig.write_image("images/Cseird_simulation_ObservedVersusSimulated(14).png")
fig.show()


# ## RMSE AND MAE

# In[7]:


observed_IRD = covid_history.loc[0:13, ['daily_active_cases','total_recovered_cases', 'total_death_cases', ]].values
days=14
print(observed_IRD.shape[0])
print(observed_IRD.shape)
print (observed_IRD)


# In[8]:


#tspan_fit_pred = np.arange(0, days, 1)
tspan_fit_pred = np.arange(0, observed_IRD.shape[0], 1) #days=observed_IRD.shape[0]
params['beta'].value = beta
params['sigma'].value = sigma 
params['gamma'].value = gamma 
params['mu'].value = mu
predicted = ode_solver(tspan_fit_pred, initial_conditions, params) #using non optimised params


# In[9]:


predicted_IRD_nonoptimised = predicted[:, 2:5]
print(predicted_IRD_nonoptimised.shape)


# In[10]:


print("Predicted MAE")
print(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 0] - observed_IRD[:days, 0])))
print(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 1] - observed_IRD[:days, 1])))
print(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 2] - observed_IRD[:days, 2])))

print("\nPredicted RMSE")
print(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 0] - observed_IRD[:days, 0])**2)))
print(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 1] - observed_IRD[:days, 1])**2)))
print(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 2] - observed_IRD[:days, 2])**2)))

#should consider to input slider to adjust the initial params (refer recent Malaysian literature)


# ## Part 2: Parameter Estimation (Optimised parameters_minimising error)

# In[11]:


def error(params, initial_conditions, tspan, data):
    sol = ode_solver(tspan, initial_conditions, params)
    return (sol[:, 2:5] - observed_IRD).ravel() #Return a contiguous flattened array.


# In[12]:


initial_conditions = [initS, initE, initI, initR, initD]
params['beta'].value = beta
params['sigma'].value = sigma
params['gamma'].value = gamma
params['mu'].value = mu
days = 14
tspan = np.arange(0, days, 1)
data = covid_history.loc[0:(days-1), ['daily_active_cases', 'total_recovered_cases', 'total_death_cases']].values


# In[13]:


data.shape


# In[14]:


params


# In[15]:


# fit model and find predicted values
result1 = minimize(error,params, args=(initial_conditions, tspan, data), method='least_squares')
result2 = minimize(error,params, args=(initial_conditions, tspan, data), method='leastsq')
result3 = minimize(error,params, args=(initial_conditions, tspan, data), method='nelder')
result4 = minimize(error,params, args=(initial_conditions, tspan, data), method='slsqp')


# In[16]:


result1.params 


# In[17]:


result2.params 


# In[18]:


result3.params 


# In[19]:


result4.params 


# In[20]:


# display fitted statistics
report_fit(result1)


# In[21]:


# display fitted statistics
report_fit(result2)


# In[22]:


# display fitted statistics
report_fit(result3)


# In[23]:


# display fitted statistics
report_fit(result4)


# In[24]:


final1 = data + result1.residual.reshape(data.shape) #observed_data+relative_error to plot y
final2 = data + result2.residual.reshape(data.shape)
final3 = data + result3.residual.reshape(data.shape)
final4 = data + result4.residual.reshape(data.shape)

#first figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=tspan, y=data[:, 0], mode='markers', marker_color='purple', name='Observed I', line = dict(dash='dot')))
fig.add_trace(go.Scatter(x=tspan, y=data[:, 1], mode='markers',marker_color='orange', name='Observed R', line = dict(dash='dot')))
fig.add_trace(go.Scatter(x=tspan, y=data[:, 2], mode='markers', marker_color='red',name='Observed D', line = dict(dash='dot')))
fig.add_trace(go.Scatter(x=tspan, y=final1[:, 0], mode='lines', line_color='purple', name='Fitted I'))
fig.add_trace(go.Scatter(x=tspan, y=final1[:, 1], mode='lines', line_color='orange',name='Fitted R'))
fig.add_trace(go.Scatter(x=tspan, y=final1[:, 2], mode='lines', line_color='red',name='Fitted D'))
fig.update_layout(title='SEIRD: Observed vs Fitted',
                       xaxis_title='Days',
                       yaxis_title='Populations',
                       title_x=0.5,
                      width=900, height=400
                     )

fig.write_image("images/Cseird_simulation_ObservedVersusFitted_least-squares (14days)).png")
fig. show()

#second figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=tspan, y=data[:, 0], mode='markers', marker_color='purple', name='Observed I', line = dict(dash='dot')))
fig.add_trace(go.Scatter(x=tspan, y=data[:, 1], mode='markers',marker_color='orange', name='Observed R', line = dict(dash='dot')))
fig.add_trace(go.Scatter(x=tspan, y=data[:, 2], mode='markers', marker_color='red',name='Observed D', line = dict(dash='dot')))
fig.add_trace(go.Scatter(x=tspan, y=final2[:, 0], mode='lines', line_color='purple', name='Fitted I'))
fig.add_trace(go.Scatter(x=tspan, y=final2[:, 1], mode='lines', line_color='orange',name='Fitted R'))
fig.add_trace(go.Scatter(x=tspan, y=final2[:, 2], mode='lines', line_color='red',name='Fitted D'))
fig.update_layout(title='SEIRD: Observed vs Fitted',
                       xaxis_title='Days',
                       yaxis_title='Populations',
                       title_x=0.5,
                      width=900, height=400
                     )

fig.write_image("images/Cseird_simulation_ObservedVersusFitted_leastsq (14days)).png")
fig. show()

#third figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=tspan, y=data[:, 0], mode='markers', marker_color='purple', name='Observed I', line = dict(dash='dot')))
fig.add_trace(go.Scatter(x=tspan, y=data[:, 1], mode='markers',marker_color='orange', name='Observed R', line = dict(dash='dot')))
fig.add_trace(go.Scatter(x=tspan, y=data[:, 2], mode='markers', marker_color='red',name='Observed D', line = dict(dash='dot')))
fig.add_trace(go.Scatter(x=tspan, y=final3[:, 0], mode='lines', line_color='purple', name='Fitted I'))
fig.add_trace(go.Scatter(x=tspan, y=final3[:, 1], mode='lines', line_color='orange',name='Fitted R'))
fig.add_trace(go.Scatter(x=tspan, y=final3[:, 2], mode='lines', line_color='red',name='Fitted D'))
fig.update_layout(title='SEIRD: Observed vs Fitted',
                       xaxis_title='Days',
                       yaxis_title='Populations',
                       title_x=0.5,
                      width=900, height=400
                     )

fig.write_image("images/Cseird_simulation_ObservedVersusFitted_nelder(14days)).png")
fig. show()

#fourth figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=tspan, y=data[:, 0], mode='markers', marker_color='purple', name='Observed I', line = dict(dash='dot')))
fig.add_trace(go.Scatter(x=tspan, y=data[:, 1], mode='markers',marker_color='orange', name='Observed R', line = dict(dash='dot')))
fig.add_trace(go.Scatter(x=tspan, y=data[:, 2], mode='markers', marker_color='red',name='Observed D', line = dict(dash='dot')))
fig.add_trace(go.Scatter(x=tspan, y=final4[:, 0], mode='lines', line_color='purple', name='Fitted I'))
fig.add_trace(go.Scatter(x=tspan, y=final4[:, 1], mode='lines', line_color='orange',name='Fitted R'))
fig.add_trace(go.Scatter(x=tspan, y=final4[:, 2], mode='lines', line_color='red',name='Fitted D'))
fig.update_layout(title='SEIRD: Observed vs Fitted',
                       xaxis_title='Days',
                       yaxis_title='Populations',
                       title_x=0.5,
                      width=900, height=400
                     )

fig.write_image("images/Cseird_simulation_ObservedVersusFitted_slsqp (14days)).png")
fig. show()


# In[25]:


tspan_fit_pred = np.arange(0, observed_IRD.shape[0], 1)

params['beta'].value = result1.params['beta'].value
params['sigma'].value = result1.params['sigma'].value
params['gamma'].value = result1.params['gamma'].value
params['mu'].value = result1.params['mu'].value
predicted1 = ode_solver(tspan_fit_pred, initial_conditions, params) #using optimised params

params['beta'].value = result2.params['beta'].value
params['sigma'].value = result2.params['sigma'].value
params['gamma'].value = result2.params['gamma'].value
params['mu'].value = result2.params['mu'].value
predicted2 = ode_solver(tspan_fit_pred, initial_conditions, params) #using optimised params

params['beta'].value = result3.params['beta'].value
params['sigma'].value = result3.params['sigma'].value
params['gamma'].value = result3.params['gamma'].value
params['mu'].value = result3.params['mu'].value
predicted3 = ode_solver(tspan_fit_pred, initial_conditions, params) #using optimised params

params['beta'].value = result4.params['beta'].value
params['sigma'].value = result4.params['sigma'].value
params['gamma'].value = result4.params['gamma'].value
params['mu'].value = result4.params['mu'].value
predicted4 = ode_solver(tspan_fit_pred, initial_conditions, params) #using optimised params


# In[27]:


predicted_IRD_optimised1 = predicted1[:, 2:5]
print(predicted_IRD_optimised1.shape)

predicted_IRD_optimised2 = predicted2[:, 2:5]
print(predicted_IRD_optimised2.shape)

predicted_IRD_optimised3 = predicted3[:, 2:5]
print(predicted_IRD_optimised3.shape)

predicted_IRD_optimised4 = predicted4[:, 2:5]
print(predicted_IRD_optimised4.shape)


# In[28]:


print("Fitting method: least-squares")
RMSE_MAE1= [["I",np.mean(np.abs(predicted_IRD_optimised1[:days, 0] - observed_IRD[:days, 0])),             np.sqrt(np.mean((predicted_IRD_optimised1[:days, 0] - observed_IRD[:days, 0])**2))],            ["R",np.mean(np.abs(predicted_IRD_optimised1[:days, 1] - observed_IRD[:days, 1])),
             np.sqrt(np.mean((predicted_IRD_optimised1[:days, 1] - observed_IRD[:days, 1])**2))], 
            ["D",np.mean(np.abs(predicted_IRD_optimised1[:days, 2] - observed_IRD[:days, 2])),\
             np.sqrt(np.mean((predicted_IRD_optimised1[:days, 2] - observed_IRD[:days, 2])**2))]
    
]
head=["Cases","RMSE", "MAE"]
print(tabulate(RMSE_MAE1, headers=head, tablefmt="grid"))
print("-----------------------------------------")

print("Fitting method: leastsqr")
RMSE_MAE2= [["I",np.mean(np.abs(predicted_IRD_optimised2[:days, 0] - observed_IRD[:days, 0])),             np.sqrt(np.mean((predicted_IRD_optimised2[:days, 0] - observed_IRD[:days, 0])**2))],            ["R",np.mean(np.abs(predicted_IRD_optimised2[:days, 1] - observed_IRD[:days, 1])),
             np.sqrt(np.mean((predicted_IRD_optimised2[:days, 1] - observed_IRD[:days, 1])**2))], 
            ["D",np.mean(np.abs(predicted_IRD_optimised2[:days, 2] - observed_IRD[:days, 2])),\
             np.sqrt(np.mean((predicted_IRD_optimised2[:days, 2] - observed_IRD[:days, 2])**2))]
    
]
head=["Cases","RMSE", "MAE"]
print(tabulate(RMSE_MAE2, headers=head, tablefmt="grid"))
print("-----------------------------------------")

print("Fitting method: nelder")
RMSE_MAE3= [["I",np.mean(np.abs(predicted_IRD_optimised3[:days, 0] - observed_IRD[:days, 0])),             np.sqrt(np.mean((predicted_IRD_optimised3[:days, 0] - observed_IRD[:days, 0])**2))],            ["R",np.mean(np.abs(predicted_IRD_optimised3[:days, 1] - observed_IRD[:days, 1])),
             np.sqrt(np.mean((predicted_IRD_optimised3[:days, 1] - observed_IRD[:days, 1])**2))], 
            ["D",np.mean(np.abs(predicted_IRD_optimised3[:days, 2] - observed_IRD[:days, 2])),\
             np.sqrt(np.mean((predicted_IRD_optimised3[:days, 2] - observed_IRD[:days, 2])**2))]
    
]
head=["Cases","RMSE", "MAE"]
print(tabulate(RMSE_MAE3, headers=head, tablefmt="grid"))
print("-----------------------------------------")

print("Fitting method: slsqp")
RMSE_MAE4= [["I",np.mean(np.abs(predicted_IRD_optimised4[:days, 0] - observed_IRD[:days, 0])),             np.sqrt(np.mean((predicted_IRD_optimised4[:days, 0] - observed_IRD[:days, 0])**2))],            ["R",np.mean(np.abs(predicted_IRD_optimised4[:days, 1] - observed_IRD[:days, 1])),
             np.sqrt(np.mean((predicted_IRD_optimised4[:days, 1] - observed_IRD[:days, 1])**2))], 
            ["D",np.mean(np.abs(predicted_IRD_optimised4[:days, 2] - observed_IRD[:days, 2])),\
             np.sqrt(np.mean((predicted_IRD_optimised4[:days, 2] - observed_IRD[:days, 2])**2))]
    
]
head=["Cases","RMSE", "MAE"]
print(tabulate(RMSE_MAE4, headers=head, tablefmt="grid"))
print("-----------------------------------------")


# In[29]:


a1=(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 0] - observed_IRD[:days, 0])))
a2=(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 1] - observed_IRD[:days, 1])))
a3=(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 2] - observed_IRD[:days, 2])))
b1=(np.mean(np.abs(predicted_IRD_optimised1[:days, 0] - observed_IRD[:days, 0])))
b2=(np.mean(np.abs(predicted_IRD_optimised1[:days, 1] - observed_IRD[:days, 1])))
b3=(np.mean(np.abs(predicted_IRD_optimised1[:days, 2] - observed_IRD[:days, 2])))
c1=(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 0] - observed_IRD[:days, 0])**2)))
c2=(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 1] - observed_IRD[:days, 1])**2)))
c3=(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 2] - observed_IRD[:days, 2])**2)))
d1=(np.sqrt(np.mean((predicted_IRD_optimised1[:days, 0] - observed_IRD[:days, 0])**2)))
d2=(np.sqrt(np.mean((predicted_IRD_optimised1[:days, 1] - observed_IRD[:days, 1])**2)))
d3=(np.sqrt(np.mean((predicted_IRD_optimised1[:days, 2] - observed_IRD[:days, 2])**2)))

print("-----------------------------------")
print("Fitting method: least-squares")
#DA=decreasedamount
DA1= [["I",((a1-b1)/(a1))*100, ((c1-d1)/(c1))*100 ],            ["R",((a2-b2)/(a2))*100, ((c2-d2)/(c2))*100], 
            ["D",((a3-b3)/(a3))*100, ((c3-d3)/(c3))*100]
    
]
head=["Cases","Decreased amount of MAE(%)", "Decreased amount of RMSE(%)"]
print(tabulate(DA1, headers=head, tablefmt="grid"))
print("-----------------------------------------")


a1=(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 0] - observed_IRD[:days, 0])))
a2=(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 1] - observed_IRD[:days, 1])))
a3=(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 2] - observed_IRD[:days, 2])))
b1=(np.mean(np.abs(predicted_IRD_optimised2[:days, 0] - observed_IRD[:days, 0])))
b2=(np.mean(np.abs(predicted_IRD_optimised2[:days, 1] - observed_IRD[:days, 1])))
b3=(np.mean(np.abs(predicted_IRD_optimised2[:days, 2] - observed_IRD[:days, 2])))
c1=(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 0] - observed_IRD[:days, 0])**2)))
c2=(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 1] - observed_IRD[:days, 1])**2)))
c3=(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 2] - observed_IRD[:days, 2])**2)))
d1=(np.sqrt(np.mean((predicted_IRD_optimised2[:days, 0] - observed_IRD[:days, 0])**2)))
d2=(np.sqrt(np.mean((predicted_IRD_optimised2[:days, 1] - observed_IRD[:days, 1])**2)))
d3=(np.sqrt(np.mean((predicted_IRD_optimised2[:days, 2] - observed_IRD[:days, 2])**2)))

print("-----------------------------------")
print("Fitting method: leastsq")
#DA=decreasedamount
DA1= [["I",((a1-b1)/(a1))*100, ((c1-d1)/(c1))*100 ],            ["R",((a2-b2)/(a2))*100, ((c2-d2)/(c2))*100], 
            ["D",((a3-b3)/(a3))*100, ((c3-d3)/(c3))*100]
    
]
head=["Cases","Decreased amount of MAE(%)", "Decreased amount of RMSE(%)"]
print(tabulate(DA1, headers=head, tablefmt="grid"))
print("-----------------------------------------")

a1=(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 0] - observed_IRD[:days, 0])))
a2=(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 1] - observed_IRD[:days, 1])))
a3=(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 2] - observed_IRD[:days, 2])))
b1=(np.mean(np.abs(predicted_IRD_optimised3[:days, 0] - observed_IRD[:days, 0])))
b2=(np.mean(np.abs(predicted_IRD_optimised3[:days, 1] - observed_IRD[:days, 1])))
b3=(np.mean(np.abs(predicted_IRD_optimised3[:days, 2] - observed_IRD[:days, 2])))
c1=(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 0] - observed_IRD[:days, 0])**2)))
c2=(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 1] - observed_IRD[:days, 1])**2)))
c3=(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 2] - observed_IRD[:days, 2])**2)))
d1=(np.sqrt(np.mean((predicted_IRD_optimised3[:days, 0] - observed_IRD[:days, 0])**2)))
d2=(np.sqrt(np.mean((predicted_IRD_optimised3[:days, 1] - observed_IRD[:days, 1])**2)))
d3=(np.sqrt(np.mean((predicted_IRD_optimised3[:days, 2] - observed_IRD[:days, 2])**2)))

print("-----------------------------------")
print("Fitting method: nelder")
#DA=decreasedamount
DA1= [["I",((a1-b1)/(a1))*100, ((c1-d1)/(c1))*100 ],            ["R",((a2-b2)/(a2))*100, ((c2-d2)/(c2))*100], 
            ["D",((a3-b3)/(a3))*100, ((c3-d3)/(c3))*100]
    
]
head=["Cases","Decreased amount of MAE(%)", "Decreased amount of RMSE(%)"]
print(tabulate(DA1, headers=head, tablefmt="grid"))
print("-----------------------------------------")

a1=(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 0] - observed_IRD[:days, 0])))
a2=(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 1] - observed_IRD[:days, 1])))
a3=(np.mean(np.abs(predicted_IRD_nonoptimised[:days, 2] - observed_IRD[:days, 2])))
b1=(np.mean(np.abs(predicted_IRD_optimised4[:days, 0] - observed_IRD[:days, 0])))
b2=(np.mean(np.abs(predicted_IRD_optimised4[:days, 1] - observed_IRD[:days, 1])))
b3=(np.mean(np.abs(predicted_IRD_optimised4[:days, 2] - observed_IRD[:days, 2])))
c1=(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 0] - observed_IRD[:days, 0])**2)))
c2=(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 1] - observed_IRD[:days, 1])**2)))
c3=(np.sqrt(np.mean((predicted_IRD_nonoptimised[:days, 2] - observed_IRD[:days, 2])**2)))
d1=(np.sqrt(np.mean((predicted_IRD_optimised4[:days, 0] - observed_IRD[:days, 0])**2)))
d2=(np.sqrt(np.mean((predicted_IRD_optimised4[:days, 1] - observed_IRD[:days, 1])**2)))
d3=(np.sqrt(np.mean((predicted_IRD_optimised4[:days, 2] - observed_IRD[:days, 2])**2)))

print("-----------------------------------")
print("Fitting method:slsqp")
#DA=decreasedamount
DA1= [["I",((a1-b1)/(a1))*100, ((c1-d1)/(c1))*100 ],            ["R",((a2-b2)/(a2))*100, ((c2-d2)/(c2))*100], 
            ["D",((a3-b3)/(a3))*100, ((c3-d3)/(c3))*100]
    
]
head=["Cases","Decreased amount of MAE(%)", "Decreased amount of RMSE(%)"]
print(tabulate(DA1, headers=head, tablefmt="grid"))
print("-----------------------------------------")


# ## Part 3: Simulation of Classical SEIRD Model (optimised constant params)

# In[30]:


#SEIRD Model simulation using optimised parameter for 14days, Simulation period: 110 days
initial_conditions = [initS, initE, initI, initR, initD]
params['beta'].value = result1.params['beta'].value
params['sigma'].value = result1.params['sigma'].value
params['gamma'].value = result1.params['gamma'].value
params['mu'].value = result1.params['mu'].value


days=110
t_fit_pred = np.arange(0, days, 1)

#solve ODE with optimized parameter
opt = ode_solver(t_fit_pred, initial_conditions, params)
S, E, I, R, D = opt[:, 0], opt[:, 1], opt[:, 2], opt[:, 3], opt[:, 4]

# Create trace 
#SEIRD model simulation using optimized parameter
fig = go.Figure()
#fig.add_trace(go.Scatter(x=t_fit_pred, y=S, mode='lines', line_color='blue', name='Susceptible'))
#fig.add_trace(go.Scatter(x=t_fit_pred, y=E, mode='lines', line_color='turquoise',name='Exposed'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=I, mode='lines', line_color='purple',name='Simulated I'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=R, mode='lines',line_color='orange',name='Simulated R'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=D, mode='lines',line_color='red',name='Simulated D'))


#observed data for 14days
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:362].daily_active_cases, mode='markers', marker_color='purple',                             name='Observed  I', line = dict(dash='dash')))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:362].total_recovered_cases, mode='markers',marker_color='orange',                             name='Observed R', line = dict(dash='dash')))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:362].total_death_cases, mode='markers', marker_color='red',                           name='Observed D', line = dict(dash='dash')))
if days <= 30:
        step = 1
elif days <= 90:
        step = 5
else:
        step = 50

# Edit the layout
fig.update_layout(title='Simulation of Classical SEIRD Model',
                       xaxis_title='Days',
                       yaxis_title='Populations',
                       title_x=0.5, font_size= 24,
                      width=900, height=600
                     )
fig.update_xaxes(tickangle=0, tickformat = None, tickmode='array', tickvals=np.arange(0, days + 1, step))
fig.write_image("images/Cseird_simulation_optimisedparams_14.png")
fig.show()


# In[38]:


#SEIRD Model simulation using optimised parameter for 14days
initial_conditions = [initS, initE, initI, initR, initD]
params['beta'].value = result1.params['beta'].value
params['sigma'].value = result1.params['sigma'].value
params['gamma'].value = result1.params['gamma'].value
params['mu'].value = result1.params['mu'].value

days=14
t_fit_pred = np.arange(0, days, 1)

#solve ODE with optimized parameter
opt = ode_solver(t_fit_pred, initial_conditions, params)
S, E, I, R, D = opt[:, 0], opt[:, 1], opt[:, 2], opt[:, 3], opt[:, 4]

# Create trace 
#SEIRD model simulation using optimized parameter
fig = go.Figure()
#fig.add_trace(go.Scatter(x=t_fit_pred, y=S, mode='lines', line_color='blue', name='Susceptible'))
#fig.add_trace(go.Scatter(x=t_fit_pred, y=E, mode='lines', line_color='turquoise',name='Exposed'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=I, mode='lines', line_color='purple',name='Simulated I'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=R, mode='lines',line_color='orange',name='Simulated R'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=D, mode='lines',line_color='red',name='Simulated D'))
#observed data for 14days
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].daily_active_cases, mode='markers', marker_color='purple',                             name='Observed  I', line = dict(dash='dash')))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].total_recovered_cases, mode='markers',marker_color='orange',                             name='Observed R', line = dict(dash='dash')))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].total_death_cases, mode='markers', marker_color='red',                           name='Observed D', line = dict(dash='dash')))
if days <= 30:
        step = 1
elif days <= 90:
        step = 5
else:
        step = 50

# Edit the layout
fig.update_layout(title='Simulation of Classical SEIRD Model',
                       xaxis_title='Days',
                       yaxis_title='Populations',
                       title_x=0.5, font_size= 24,
                      width=900, height=600
                     )
fig.update_xaxes(tickangle=0, tickformat = None, tickmode='array', tickvals=np.arange(0, days + 1, step))
fig.write_image("images/Cseird_simulation_optimisedparams_least-squares_14.png")
fig.show()

params['beta'].value = result2.params['beta'].value
params['sigma'].value = result2.params['sigma'].value
params['gamma'].value = result2.params['gamma'].value
params['mu'].value = result2.params['mu'].value

#solve ODE with optimized parameter
opt = ode_solver(t_fit_pred, initial_conditions, params)
S, E, I, R, D = opt[:, 0], opt[:, 1], opt[:, 2], opt[:, 3], opt[:, 4]

# Create trace 
#SEIRD model simulation using optimized parameter
fig = go.Figure()
#fig.add_trace(go.Scatter(x=t_fit_pred, y=S, mode='lines', line_color='blue', name='Susceptible'))
#fig.add_trace(go.Scatter(x=t_fit_pred, y=E, mode='lines', line_color='turquoise',name='Exposed'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=I, mode='lines', line_color='purple',name='Simulated I'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=R, mode='lines',line_color='orange',name='Simulated R'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=D, mode='lines',line_color='red',name='Simulated D'))


#observed data for 14days
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].daily_active_cases, mode='markers', marker_color='purple',                             name='Observed  I', line = dict(dash='dash')))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].total_recovered_cases, mode='markers',marker_color='orange',                             name='Observed R', line = dict(dash='dash')))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].total_death_cases, mode='markers', marker_color='red',                           name='Observed D', line = dict(dash='dash')))
if days <= 30:
        step = 1
elif days <= 90:
        step = 5
else:
        step = 50

# Edit the layout
fig.update_layout(title='Simulation of Classical SEIRD Model',
                       xaxis_title='Days',
                       yaxis_title='Populations',
                       title_x=0.5, font_size= 24,
                      width=900, height=600
                     )
fig.update_xaxes(tickangle=0, tickformat = None, tickmode='array', tickvals=np.arange(0, days + 1, step))
fig.write_image("images/Cseird_simulation_optimisedparams_leastsq_14.png")
fig.show()

params['beta'].value = result3.params['beta'].value
params['sigma'].value = result3.params['sigma'].value
params['gamma'].value = result3.params['gamma'].value
params['mu'].value = result3.params['mu'].value

#solve ODE with optimized parameter
opt = ode_solver(t_fit_pred, initial_conditions, params)
S, E, I, R, D = opt[:, 0], opt[:, 1], opt[:, 2], opt[:, 3], opt[:, 4]

# Create trace 
#SEIRD model simulation using optimized parameter
fig = go.Figure()
#fig.add_trace(go.Scatter(x=t_fit_pred, y=S, mode='lines', line_color='blue', name='Susceptible'))
#fig.add_trace(go.Scatter(x=t_fit_pred, y=E, mode='lines', line_color='turquoise',name='Exposed'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=I, mode='lines', line_color='purple',name='Simulated I'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=R, mode='lines',line_color='orange',name='Simulated R'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=D, mode='lines',line_color='red',name='Simulated D'))


#observed data for 14days
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].daily_active_cases, mode='markers', marker_color='purple',                             name='Observed  I', line = dict(dash='dash')))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].total_recovered_cases, mode='markers',marker_color='orange',                             name='Observed R', line = dict(dash='dash')))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].total_death_cases, mode='markers', marker_color='red',                           name='Observed D', line = dict(dash='dash')))
if days <= 30:
        step = 1
elif days <= 90:
        step = 5
else:
        step = 50

# Edit the layout
fig.update_layout(title='Simulation of Classical SEIRD Model',
                       xaxis_title='Days',
                       yaxis_title='Populations',
                       title_x=0.5, font_size= 24,
                      width=900, height=600
                     )
fig.update_xaxes(tickangle=0, tickformat = None, tickmode='array', tickvals=np.arange(0, days + 1, step))
fig.write_image("images/Cseird_simulation_optimisedparams_nelder_14.png")
fig.show()

params['beta'].value = result4.params['beta'].value
params['sigma'].value = result4.params['sigma'].value
params['gamma'].value = result4.params['gamma'].value
params['mu'].value = result4.params['mu'].value

#solve ODE with optimized parameter
opt = ode_solver(t_fit_pred, initial_conditions, params)
S, E, I, R, D = opt[:, 0], opt[:, 1], opt[:, 2], opt[:, 3], opt[:, 4]

# Create trace 
#SEIRD model simulation using optimized parameter
fig = go.Figure()
#fig.add_trace(go.Scatter(x=t_fit_pred, y=S, mode='lines', line_color='blue', name='Susceptible'))
#fig.add_trace(go.Scatter(x=t_fit_pred, y=E, mode='lines', line_color='turquoise',name='Exposed'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=I, mode='lines', line_color='purple',name='Simulated I'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=R, mode='lines',line_color='orange',name='Simulated R'))
fig.add_trace(go.Scatter(x=t_fit_pred, y=D, mode='lines',line_color='red',name='Simulated D'))


#observed data for 14days
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].daily_active_cases, mode='markers', marker_color='purple',                             name='Observed  I', line = dict(dash='dash')))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].total_recovered_cases, mode='markers',marker_color='orange',                             name='Observed R', line = dict(dash='dash')))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].total_death_cases, mode='markers', marker_color='red',                           name='Observed D', line = dict(dash='dash')))
if days <= 30:
        step = 1
elif days <= 90:
        step = 5
else:
        step = 50

# Edit the layout
fig.update_layout(title='Simulation of Classical SEIRD Model',
                       xaxis_title='Days',
                       yaxis_title='Populations',
                       title_x=0.5, font_size= 24,
                      width=900, height=600
                     )
fig.update_xaxes(tickangle=0, tickformat = None, tickmode='array', tickvals=np.arange(0, days + 1, step))
fig.write_image("images/Cseird_simulation_optimisedparams_slsqp_14.png")
fig.show()


# In[39]:


#SEIRD Model simulation for each individual cases IRD using optimized parameter for 14days
initial_conditions = [initS, initE, initI, initR, initD]
params['beta'].value = result1.params['beta'].value
params['sigma'].value = result1.params['sigma'].value
params['gamma'].value = result1.params['gamma'].value
params['mu'].value = result1.params['mu'].value


days=14
t_fit_pred = np.arange(0, days, 1)

#solve ODE with optimized parameter
opt = ode_solver(t_fit_pred, initial_conditions, params)
S, E, I, R, D = opt[:, 0], opt[:, 1], opt[:, 2], opt[:, 3], opt[:, 4]

# Create trace 
#SEIRD model simulation using optimized parameter
fig = go.Figure()
fig.add_trace(go.Scatter(x=t_fit_pred, y=I, mode='lines', line_color='purple',name='Simulated I'))
#observed data for 14days
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].daily_active_cases, mode='markers', marker_color='purple',                             name='Observed  I', line = dict(dash='dash')))

if days <= 30:
        step = 1
elif days <= 90:
        step = 5
else:
        step = 50

# Edit the layout
fig.update_layout(title='Simulation of Classical SEIRD Model',
                       xaxis_title='Days',
                       yaxis_title='Populations',
                       title_x=0.5, font_size= 24,
                      width=900, height=600
                     )
fig.update_xaxes(tickangle=0, tickformat = None, tickmode='array', tickvals=np.arange(0, days + 1, step))
fig.write_image("images/Cseird_simulation_optimisedparams_I_14.png")
fig.show()


# In[40]:


#SEIRD Model simulation for each individual cases IRD using optimized parameter for 14days
initial_conditions = [initS, initE, initI, initR, initD]
params['beta'].value = result1.params['beta'].value
params['sigma'].value = result1.params['sigma'].value
params['gamma'].value = result1.params['gamma'].value
params['mu'].value = result1.params['mu'].value


days=14
t_fit_pred = np.arange(0, days, 1)

#solve ODE with optimized parameter
opt = ode_solver(t_fit_pred, initial_conditions, params)
S, E, I, R, D = opt[:, 0], opt[:, 1], opt[:, 2], opt[:, 3], opt[:, 4]

# Create trace 
#SEIRD model simulation using optimized parameter
fig = go.Figure()
fig.add_trace(go.Scatter(x=t_fit_pred, y=R, mode='lines',line_color='orange',name='Simulated R'))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].total_recovered_cases, mode='markers',marker_color='orange',                             name='Observed R', line = dict(dash='dash')))

if days <= 30:
        step = 1
elif days <= 90:
        step = 5
else:
        step = 50

# Edit the layout
fig.update_layout(title='Simulation of Classical SEIRD Model',
                       xaxis_title='Days',
                       yaxis_title='Populations',
                       title_x=0.5, font_size= 24,
                      width=900, height=600
                     )
fig.update_xaxes(tickangle=0, tickformat = None, tickmode='array', tickvals=np.arange(0, days + 1, step))
fig.write_image("images/Cseird_simulation_optimisedparams_R_14.png")
fig.show()


# In[41]:


#SEIRD Model simulation for each individual cases IRD using optimized parameter for 14days
initial_conditions = [initS, initE, initI, initR, initD]
params['beta'].value = result1.params['beta'].value
params['sigma'].value = result1.params['sigma'].value
params['gamma'].value = result1.params['gamma'].value
params['mu'].value = result1.params['mu'].value


days=14
t_fit_pred = np.arange(0, days, 1)

#solve ODE with optimized parameter
opt = ode_solver(t_fit_pred, initial_conditions, params)
S, E, I, R, D = opt[:, 0], opt[:, 1], opt[:, 2], opt[:, 3], opt[:, 4]

# Create trace 
#SEIRD model simulation using optimized parameter
fig = go.Figure()
fig.add_trace(go.Scatter(x=t_fit_pred, y=D, mode='lines',line_color='red',name='Simulated D'))
fig.add_trace(go.Scatter(x=tspan, y=covid_history.iloc[0:14].total_death_cases, mode='markers', marker_color='red',                           name='Observed D', line = dict(dash='dash')))
if days <= 30:
        step = 1
elif days <= 90:
        step = 5
else:
        step = 50

# Edit the layout
fig.update_layout(title='Simulation of Classical SEIRD Model',
                       xaxis_title='Days',
                       yaxis_title='Populations',
                       title_x=0.5, font_size= 24,
                      width=900, height=600
                     )
fig.update_xaxes(tickangle=0, tickformat = None, tickmode='array', tickvals=np.arange(0, days + 1, step))
fig.write_image("images/Cseird_simulation_optimisedparams_D_14.png")
fig.show()

