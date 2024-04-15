from re import L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from torch import optim
from torch.utils import data
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

import statistics
import datetime
import os
import csv
import math
import time
import numpy as np
import os

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Record the start time
start_time = time.time()

os.getcwd()

start = time.time()

idvg_temp = pd.read_csv(r'./csvData/PBTI_Variation.csv', encoding='utf8')

vov = [0.5, 1.0, 1.7]
vd_temp=[0.01, 0.012, 0.015, 0.018, 0.022, 0.027, 0.034, 0.041, 0.050, 0.062, 0.075, 0.092, 0.113, 0.138, 0.169, 0.207, 0.253, 0.310, 0.379, 0.464, 0.568, 0.695, 0.851, 1.042, 1.275, 1.560, 1.9600, 2.337, 2.86, 3.4]
temperature = np.array([258.15, 298.15, 358.15]) # in Kelvin
stress_time = np.array([1, 4, 7, 10, 100, 200, 400, 600, 1000])
vd = np.array(vd_temp)
vg_temp=idvg_temp.iloc[:,0]
vg = np.array(vg_temp.values)
vov = np.array(vov)

def Logset(target):
    temp = np.array(target)
    temp = np.log10(temp)
    return temp

It = []
for t in list(range(len(temperature))):
    for s in list(range(len(stress_time))):
        for l in list(range(len(vov))):
            for i in list(range(len(vd))):
                col_index = 2*i + 1 + 2*len(vd)*l + 2*len(vd)*len(vov)*s + 2*len(vd)*len(vov)*len(stress_time)*t
                temp = idvg_temp.iloc[:, col_index]
                temp = np.array(temp.values)
                It.extend(temp)

It = Logset(It)
# vd = Logset(vd)

def normaliz(target):
    Min = min(target)
    Val = target - Min
    Max = max(Val)
    if Max == 0:
        Norm = 1
    else:
        Norm = 1 / Max
    return (Norm, Val, Min)

(normVg, Vg_1, MinVg)=normaliz(vg)
(normVd, Vd_1, MinVd)=normaliz(vd)
(normIt, It_1, MinIt)=normaliz(It)
(normvov, vov_1, Minvov) = normaliz(vov)
(normtemperature, temperature_1, Mintemperature) = normaliz(temperature)
(normstress_time, stress_time_1, Minstress_time) = normaliz(stress_time)

Vg = normVg*Vg_1
Vd = normVd*Vd_1
I = normIt*It_1
Vov = normvov*vov_1
temperature = normtemperature*temperature_1
stress_time = normstress_time*stress_time_1

datasets = []
for t in list(range(len(temperature))):
    for s in list(range(len(stress_time))):
        for l in list(range(len(Vov))):
            for i in list(range(len(vd))):
                for j in list(range(len(vg))):
                    index = j + len(vg) * (i + len(vd) * (l + len(Vov) * (s + len(stress_time) * t)))
                    temp = [vg[j], vd[i], Vov[l], temperature[t], stress_time[s], I[index]]
                    datasets.append(temp)

V = []
for i in list(range(len(datasets))):
    temp = [datasets[i][0], datasets[i][1], datasets[i][2], datasets[i][3], datasets[i][4]]
    V.append(temp)

I = []
for i in list(range(len(datasets))):
    temp = [datasets[i][5]]
    I.append(temp)

V = torch.tensor(V)
I = torch.tensor(I)

# dataset = list(zip(V, I))
x_train, x_test, y_train, y_test = train_test_split(V, I, test_size=0.1, random_state=41)
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size = 64, num_workers = 20, shuffle=True)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=23, pin_memory=True)
testdataloader = DataLoader(TensorDataset(x_test, y_test))

# print(idvg_temp.values)
It_g = [10**x for x in It]
CM_git = np.corrcoef(vg,It_g[len(vg)*10:len(vg)*11])
Itd=[]
print(len(It_g))
print(len(vg))
print(len(vd))
print(len(vov))

for i in list(range(len(vd))):
    Itd.append(It_g[len(vg)-23+len(vg)*i])
print(Itd)
CM_dit = np.corrcoef(vd, Itd)

print(vov)
print()
Itl = []
print(It_g[len(vg)*len(vd)-4])
print(It_g[len(vg)*len(vd)*8-4])
print(list(range(len(vov))))
for i in list(range(len(vov))):
    Itl.append(It[len(vg)*len(vd)*(i+1)-20] )
print(Itl)
CM_lit = np.corrcoef(Vov, Itl)

print(CM_git)
print(CM_dit)
print(CM_lit)

n1 = 40 #40
n2 = 20 #20

# Define the neural network class
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(5, n1)
        self.fc2 = torch.nn.Linear(n1, n2)
        self.fc3 = torch.nn.Linear(n2, 1)
        self.dropout = torch.nn.Dropout(0.1)
        self.tanh = torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(n1)
        self.bn2 = torch.nn.BatchNorm1d(n2)
        self.bn3 = torch.nn.BatchNorm1d(1)

    def forward(self, x):
        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.tanh(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        x = self.tanh(x)
        #x = self.dropout(x)
        x = self.fc3(x)
        #x = self.bn3(x)
        return x

# Create an instance of the MLP class
model = MLP()
"""
# Wrap the model with nn.DataParallel to use multiple GPUs
if torch.cuda.is_available():
    model = model.cuda()
    model = nn.DataParallel(model)
"""
def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

model.apply(initialize_weights)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
#torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# scaler = GradScaler()
print('Training process has started.')
nb_epochs = 100
MLoss = []
for epoch in range(0, nb_epochs):

    current_loss = 0.0
    losses = []
    # Iterate over the dataloader for training data
    for i, data in enumerate(dataloader, 0):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0],1))

        #zero the gradients
        optimizer.zero_grad()

        #perform forward pass
        outputs = model(inputs)
        L_weight = 3
        #compute loss
        batch_loss = []
        for j in range(inputs.size(0)):
            input_j = inputs[j].reshape((1, inputs.shape[1]))
            if input_j[0,0]>0.3:
                batch_loss.append(L_weight*loss_function(outputs[j], targets[j]))
            else:
                batch_loss.append(loss_function(outputs[j], targets[j]))

        loss = torch.stack(batch_loss).mean()

        losses.append(loss.item())

        #perform backward pass
        loss.backward()
        #perform optimization
        optimizer.step()
        # Print statistics

    mean_loss = sum(losses)/len(losses)
    scheduler.step(mean_loss)

    print('Loss (epoch: %4d): %.8f' %(epoch+1, mean_loss))
# Print the loss only every 10 epochs
    #if (epoch + 1) % 10 == 0:
    #    print('Loss (epoch: %4d): %.8f' % (epoch + 1, mean_loss))
    current_loss = 0.0
    MLoss.append(mean_loss)

    #optimizer.step()
        # Print statistics
    #mean_loss = sum(losses) / len(losses)
    #scheduler.step(mean_loss)


# Process is complete.
print('Training process has finished.')

torch.save(model, 'IWO_idvg.pt')
torch.save(model.state_dict(), 'IWO_idvg_state_dict.pt')

####### loss vs. epoch #######
xloss = list(range(0, nb_epochs))
plt.plot(xloss, np.log10(MLoss))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
"""
Matplotlib fit plot
"""
weights_1 = model.fc1.weight.detach().numpy()
bias_1 = model.fc1.bias.detach().numpy()
weights_2 = model.fc2.weight.detach().numpy()
bias_2 = model.fc2.bias.detach().numpy()
weights_3 = model.fc3.weight.detach().numpy()
bias_3 = model.fc3.bias.detach().numpy()

verilog_code = ""

# Create the Verilog-A code for the 1st hidden layer
verilog_code = "real "
verilog_code += ", ".join([f"h1_{i}" for i in range(n1)]) + ";\n"

for i in range(n1):
    inputs = ["Vgs", "Vds", "Lg", "ts", "temp"]
    inputs = ["*".join([str(weights_1[i][j]), inp]) for j, inp in enumerate(inputs)]
    inputs = "+".join(inputs)
    inputs = "+".join([inputs, str(bias_1[i])])
    verilog_code += "h1_{} = tanh({});\n".format(i, inputs)

# Create the Verilog-A code for the 2nd hidden layer
verilog_code += "real "
verilog_code += ", ".join([f"h2_{i}" for i in range(n2)]) + ";\n"
for i in range(n2):
    inputs = ["h1_{}".format(j) for j in range(n1)]
    inputs = ["*".join([str(weights_2[i][j]), inp]) for j, inp in enumerate(inputs)]
    inputs = "+".join(inputs)
    inputs = "+".join([inputs, str(bias_2[i])])
    verilog_code += "h2_{} = tanh({});\n".format(i, inputs)

# Create the Verilog-A code for the output layer
inputs = ["h2_{}".format(i) for i in range(n2)]
inputs = ["*".join([str(weights_3[0][i]), inp]) for i, inp in enumerate(inputs)]
inputs = "+".join(inputs)
inputs = "+".join([inputs, str(bias_3[0])])
verilog_code += "y = {};\n".format(inputs)

verilog_code = """
module IWO_verilogA (d, g, s);
	inout d, g, s;
	electrical d, g, s;
	
	//****** Parameters L and W ********
	parameter real W = 0.1; //set on cadence
	parameter real L = 0.05; //set on cadence
	parameter real vov = 1.7; //set on cadence, 0.5, 1.0, 1.7
	parameter real Temp = 85; //set on cadence
	parameter real StressTime = 100; //set on cadence
	parameter MinVg = {} ;
	parameter normVg = {} ;
	parameter MinVd = {} ;
	parameter normVd = {} ;
	parameter Minvov = {} ;
	parameter normvov = {} ;
	parameter MinI = {} ;
	parameter normI = {};
	parameter MinO = 8.15e-15 ;
	parameter normO =33613445378151.26;
	parameter Mintemp = {};
	parameter normtemp = {};
	parameter Mints = {};
	parameter normts = {};

	real Vg, Vd, Vs, Vgs, Vds, Id, Cgg, Cgsd, Vgd;
	real Vgsraw, Vgdraw, dir;
	real Vov, ts, temp; // BTI variable: Vov, stress time, temperature

analog begin
	Vg = V(g);
	Vs = V(s);
	Vd = V(d);
    Vgsraw = Vg-Vs ;
    Vgdraw = Vg-Vd ;
if (Vgsraw>=Vgdraw) begin
	Vgs = ((Vg-Vs) - MinVg) * normVg ;
    dir = 1 ;
end
else begin
	Vgs = ((Vg-Vd) - MinVg) * normVg ;
    dir = -1 ;
end

	Vds = (abs(Vd-Vs) - MinVd) * normVd ;
	Vov = (vov - Minvov)*normvov ;
	temp = (Temp - Mintemp)*normtemp ;
	ts = (StressTime - Mints)*normts ;

//******************** C-V NN **********************************//
hc1_0 = tanh(-0.99871427*Vgs+-0.16952373*Vds+0.32118186*Lg+0.41485423);
hc1_1 = tanh(0.31587568*Vgs+0.13397887*Vds+-0.4541538*Lg+-0.3630942);
hc1_2 = tanh(-0.76281804*Vgs+0.09352969*Vds+1.1961353*Lg+0.3904977);
hc1_3 = tanh(-1.115087*Vgs+0.85752946*Vds+-0.11746484*Lg+0.5500279);
hc1_4 = tanh(1.0741386*Vgs+0.82041687*Vds+0.19092631*Lg+-0.4009425);
hc1_5 = tanh(-0.47921795*Vgs+-0.8749933*Vds+-0.054768667*Lg+0.62785167);
hc1_6 = tanh(0.5449184*Vgs+-4.409165*Vds+-0.072947875*Lg+-0.31324536);
hc1_7 = tanh(-2.9224303*Vgs+2.7675478*Vds+0.08862238*Lg+0.6493558);
hc1_8 = tanh(0.65050656*Vgs+-0.29751927*Vds+0.1571876*Lg+-0.38088372);
hc1_9 = tanh(-0.30384183*Vgs+0.5649165*Vds+2.6806898*Lg+0.3197917);
hc1_10 = tanh(-0.095988505*Vgs+2.0158541*Vds+-0.42972717*Lg+-0.30388466);
hc1_11 = tanh(6.7699738*Vgs+-0.07234483*Vds+-0.013545353*Lg+-1.3694142);
hc1_12 = tanh(-0.3404029*Vgs+0.0443459*Vds+0.89597*Lg+0.069993004);
hc1_13 = tanh(0.62300175*Vgs+-0.29515797*Vds+1.6753465*Lg+-0.6520838);
hc1_14 = tanh(0.37957156*Vgs+0.2237372*Vds+0.08591952*Lg+0.13126835);
hc1_15 = tanh(0.19949242*Vgs+-0.26481664*Vds+-0.41059187*Lg+-0.40832308);
hc1_16 = tanh(0.98966587*Vgs+-0.24259183*Vds+0.36584845*Lg+-0.8024042);

hc2_0 = tanh(-0.91327864*hc1_0+0.4696781*hc1_1+0.3302202*hc1_2+0.11393868*hc1_3+0.45070222*hc1_4+-0.2894044*hc1_5+0.55066*hc1_6+-1.6242687*hc1_7+-0.38140613*hc1_8+0.032771554*hc1_9+0.17647126);
hc2_1 = tanh(-1.1663305*hc1_0+-0.523984*hc1_1+-0.90804136*hc1_2+-0.7418044*hc1_3+1.456171*hc1_4+-0.16802542*hc1_5+0.8235596*hc1_6+-2.2246246*hc1_7+-0.40805355*hc1_8+0.7207601*hc1_9+0.4729169);
hc2_2 = tanh(-0.69065374*hc1_0+0.40205315*hc1_1+-0.49410668*hc1_2+0.8681325*hc1_3+0.471351*hc1_4+0.46939445*hc1_5+0.45568785*hc1_6+-0.92935294*hc1_7+-0.8209646*hc1_8+0.1158967*hc1_9+-0.075798474);
hc2_3 = tanh(0.11535446*hc1_0+-0.06296927*hc1_1+-0.6740435*hc1_2+0.7428315*hc1_3+0.05890677*hc1_4+0.9579441*hc1_5+-0.037319*hc1_6+-0.18491426*hc1_7+-0.02981994*hc1_8+0.038347963*hc1_9+0.039531134);
hc2_4 = tanh(0.37817463*hc1_0+-0.6811279*hc1_1+1.1388369*hc1_2+0.19983096*hc1_3+-0.20415118*hc1_4+1.3022176*hc1_5+0.22571652*hc1_6+0.1690611*hc1_7+-0.56475276*hc1_8+-0.4069731*hc1_9+0.99962974);
hc2_5 = tanh(0.032873478*hc1_0+-0.05209407*hc1_1+-0.010839908*hc1_2+-0.13892579*hc1_3+-0.050480977*hc1_4+0.0127089145*hc1_5+0.0052771433*hc1_6+0.02029242*hc1_7+-0.08705659*hc1_8+0.0254766*hc1_9+0.025135752);
hc2_6 = tanh(-0.34639204*hc1_0+0.06937975*hc1_1+0.18671949*hc1_2+-0.18912783*hc1_3+0.1312504*hc1_4+0.4627272*hc1_5+-0.42590702*hc1_6+-0.10966313*hc1_7+0.66083515*hc1_8+-0.050718334*hc1_9+0.08234678);
hc2_7 = tanh(0.90656275*hc1_0+-0.037281644*hc1_1+0.77237594*hc1_2+1.4710428*hc1_3+0.13597831*hc1_4+-0.059844542*hc1_5+-0.7801535*hc1_6+3.7814677*hc1_7+-0.5976644*hc1_8+0.2721995*hc1_9+0.023777716);
hc2_8 = tanh(0.39720625*hc1_0+-0.45262313*hc1_1+0.19873238*hc1_2+0.9750888*hc1_3+-0.9427992*hc1_4+0.4487432*hc1_5+-0.3372945*hc1_6+0.33729544*hc1_7+-0.1667088*hc1_8+-0.5707525*hc1_9+0.27954483);
hc2_9 = tanh(0.28551984*hc1_0+-0.68350387*hc1_1+0.9916423*hc1_2+-0.8254094*hc1_3+0.09875706*hc1_4+0.47609732*hc1_5+-0.058662917*hc1_6+0.09181381*hc1_7+0.09592329*hc1_8+1.3940467*hc1_9+0.3865768);
hc2_10 = tanh(0.098737516*hc1_0+0.060473576*hc1_1+0.42824662*hc1_2+0.15018038*hc1_3+0.082621895*hc1_4+-0.00019039502*hc1_5+-0.3321634*hc1_6+0.7936295*hc1_7+-0.041197542*hc1_8+0.6530619*hc1_9+0.1338804);
hc2_11 = tanh(-0.3585284*hc1_0+-0.09956017*hc1_1+0.17224246*hc1_2+-0.016925728*hc1_3+-0.46462816*hc1_4+-0.5649022*hc1_5+1.251695*hc1_6+-0.4303161*hc1_7+0.48546878*hc1_8+0.22958975*hc1_9+-0.27899802);
hc2_12 = tanh(0.8565631*hc1_0+-0.7622999*hc1_1+0.32367912*hc1_2+1.4776785*hc1_3+0.2712369*hc1_4+0.2275511*hc1_5+0.39908803*hc1_6+4.2305493*hc1_7+-0.3467536*hc1_8+0.41231114*hc1_9+0.47123823);
hc2_13 = tanh(-0.26411077*hc1_0+-0.17583853*hc1_1+0.045439184*hc1_2+-0.13801138*hc1_3+0.03278085*hc1_4+-0.45625108*hc1_5+-0.17905861*hc1_6+0.3060186*hc1_7+-0.3361926*hc1_8+-0.050055273*hc1_9+0.17996444);
hc2_14 = tanh(0.016094366*hc1_0+-0.013196736*hc1_1+0.4124856*hc1_2+0.22926371*hc1_3+-0.35071182*hc1_4+-0.34217188*hc1_5+-0.69466996*hc1_6+1.0563152*hc1_7+-0.18019852*hc1_8+0.061871335*hc1_9+0.09762555);
hc2_15 = tanh(-0.18970782*hc1_0+0.3624813*hc1_1+-1.3419824*hc1_2+0.103635244*hc1_3+-0.14595217*hc1_4+-0.1899393*hc1_5+0.176524*hc1_6+-0.4428012*hc1_7+-0.39544868*hc1_8+-0.45783517*hc1_9+0.1884755);
hc2_16 = tanh(-0.1585831*hc1_0+0.035894837*hc1_1+-0.14261873*hc1_2+0.25914294*hc1_3+0.040607046*hc1_4+0.11555795*hc1_5+0.0022548323*hc1_6+-0.002149359*hc1_7+0.07067297*hc1_8+0.019578662*hc1_9+0.16657573);
yc = 0.17503543*hc2_0+-0.15556754*hc2_1+-0.125455*hc2_2+-0.07502612*hc2_3+-0.16671115*hc2_4+0.29854503;

Cgg = (yc / normO + MinO)*W;
Cgsd = Cgg/2 ;

//******************** I-V NN **********************************//
{}

	Id = pow(10, (y/normI + MinI))*W;

if (Id <= 1e-15) begin //limit
	Id = 1e-15;
	//Id = Id;
end
else begin
	Id = Id;
end  //limit end

	I(g, d) <+ Cgsd*ddt(Vg-Vd) ;
	I(g, s) <+ Cgsd*ddt(Vg-Vs) ;

if (Vgsraw >= Vgdraw) begin
	I(d, s) <+ dir*Id*W ;

end

else begin
	I(d, s) <+ dir*Id*W ;

end

end

endmodule


""".format(MinVg, normVg, MinVd, normVd, Minvov, normvov, MinIt, normIt, Mintemperature, normtemperature, Minstress_time, normstress_time, verilog_code)

print(verilog_code)

with open("LVIWO_BTI.va", "w") as f:
    f.write(verilog_code)

# Record the end time
end_time = time.time()

# Calculate the total time taken
time_taken = end_time - start_time

print(f"Time taken: {time_taken} seconds")
