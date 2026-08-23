import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import sys
import csv
from collections import defaultdict
from matplotlib import pyplot
import random

pyplot.rcParams['savefig.dpi'] = 1200


def exponential_smoothing(series, alpha):
    # first value is same as series
    result = [series[0]]
    for value in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[value - 1])
    return result


def consistent_name(name):
    name = name.replace('-ALL', '').replace('Mentions-', '') \
        .replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
    
    # special case
    if 'HIDDEN MARKOV MODEL' in name:
        return 'Statistical HMM'

    if name in {'CAPTCHA', 'DNSSEC', 'RRAM'}:
        return name

    # e.g., University of london
    if not name.isupper():
        words = name.split(' ')
        result = ''

        for i, word in enumerate(words):
            if len(word) <= 2:  # e.g., "of"
                result += word
            else:
                result += word[0].upper() + word[1:]
            
            if i < len(words) - 1:
                result += ' '
        return result

    words = name.split(' ')
    result = ''

    for i, word in enumerate(words):
        if len(word) <= 3 or '/' in word or word in {'MITM', 'SIEM'}:
            result += word
        else:
            result += word[0] + (word[1:].lower())
        
        if i < len(words) - 1:
            result += ' '
    return result


def zero_negative_curves(data, forecast, s):
    """
    Negative values (due to smoothing) are changed to 0
    """
    a = data[:, index[s]]
    f = forecast[:, index[s]]

    for i in range(a.shape[0]):
        if a[i] < 0:
            a[i] = 0

    for i in range(f.shape[0]):
        if f[i] < 0:
            f[i] = 0
    return data, forecast
           

def plot_forecast(data,forecast,confidence,s,index,col):
    """
    Plot past data and the forecast of a single pertinent technology node s
    """
    data, forecast = zero_negative_curves(data, forecast, s)

    pyplot.style.use("seaborn-dark") 
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Plot the forecast & connect the past to future in the plot
    d = torch.cat((data[:, index[s]],forecast[0:1, index[s]]), dim=0)
    f = forecast[:, index[s]]
    c = confidence[:, index[s]]
    s = consistent_name(s)

    # past
    ax.plot(range(len(d)), d, '-', color='red', label=s, linewidth=1)

    # future
    ax.plot(
        range(len(d) - 1, (len(d) + len(f)) - 1), f,
        '-', color='red', linewidth=1)

    ax.fill_between(
        range(len(d) - 1, (len(d) + len(f)) - 1),
        f - c, f + c, color='red', alpha=0.6
    )

    x = ['2012', '2013','2014', '2015', '2016', '2017', '2018', '2019',
         '2020', '2021', '2022', '2023', '2024', '2025', '2026']

    # positions of years on x axis
    ax.set_xticks(
        [6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126, 138, 150, 162, 174], x)

    ax.set_ylabel("Trend", fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.axis('tight')
    ax.grid(True)

    pyplot.xticks(rotation=90, fontsize=13)
    pyplot.title(s, y=1.03, fontsize=18)

    fig = pyplot.gcf()
    fig.set_size_inches(10, 7) 

    # Save & show the forecast
    images_dir = 'model/Bayesian/forecast/pt_plots/'

    # Save PNG
    png_path = images_dir + s.replace('/', '_') + '.png'
    pyplot.savefig(png_path, bbox_inches="tight")

    # Save PDF
    pdf_path = images_dir + s.replace('/', '_') + ".pdf"
    pyplot.savefig(pdf_path, bbox_inches="tight", format='pdf')

    pyplot.show(block=False)
    pyplot.pause(5)
    pyplot.close()


def create_columns(file_name):
    """
    Given a data file, returns the list of column names and dictionary of
    the format (column name,column index)
    """
    col_index = {}

    # Read the CSV file of the dataset
    with open(file_name, 'r') as f:
        reader = csv.reader(f)

        # Read the first row
        col_name = [c for c in next(reader)]
        if 'Date' in col_name[0]:
            col_name= col_name[1:]
        
        for i, c in enumerate(col_name):
            col_index[c] = i
        
        return col_name, col_index


def build_graph(file_name):
    """
    Build the attacks and pertinent technologies graph
    """
    # Initialise an empty dictionary with default value as an empty list
    graph = defaultdict(list)

    # Read the graph CSV file
    with open(file_name, 'r') as f:
        reader = csv.reader(f)

        # Iterate over each row in the CSV file
        for row in reader:
            # Extract the key node from the first column
            key_node = row[0]

            # Extract adjacent nodes from remaining columns (not empty columns)
            adjacent_nodes = [node for node in row[1:] if node]
            
            # Add the adjacent nodes to the graph dictionary
            graph[key_node].extend(adjacent_nodes)
    print('Graph loaded with',len(graph),'attacks...')
    return graph


# This script forecasts the future of the graph, up to 3 years in advance
data_file = './data/sm_data.txt'
model_file = 'model/Bayesian/o_model.pt'
nodes_file = 'data/data.csv'
graph_file = 'data/graph.csv'

# Read the data
fin = open(data_file)
rawdat = np.loadtxt(fin, delimiter='\t')
n, m = rawdat.shape

# Load column names and dictionary of (column name, index)
col, index=create_columns(nodes_file)

#build the graph in the format {attack:list of pertinent technologies}
graph=build_graph(graph_file)

# For normalisation
scale = np.ones(m)
dat = np.zeros(rawdat.shape)

# Normalise
for i in range(m):
    scale[i] = np.max(np.abs(rawdat[:, i]))
    dat[:, i] = rawdat[:, i] / np.max(np.abs(rawdat[:, i]))

print('data shape:', dat.shape)

# Pprepe the last part of the data for the forecast
P = 10  # look back

X = torch.from_numpy(dat[-P:, :])  # Look back 10 months
X = torch.unsqueeze(X, dim=0)
X = torch.unsqueeze(X, dim=1)
X = X.transpose(2, 3)
X = X.to(torch.float)

# Load the model
model = None
with open(model_file, 'rb') as f:
    model = torch.load(f)

# Bayesian estimation
num_runs = 10

# Create a list to store the outputs
outputs = []

# Use model to predict next time step
for _ in range(num_runs):
    with torch.no_grad():
        output = model(X)  
        y_pred = output[-1, :, :, -1].clone()  # 36x142
    outputs.append(y_pred)

# Stack the outputs along a new dimension
outputs = torch.stack(outputs)

# Variance and Standard Deviation
Y = torch.mean(outputs, dim=0)
variance = torch.var(outputs, dim=0)
std_dev = torch.std(outputs, dim=0)

# Calculate 95% confidence interval
z = 1.96
confidence = z * std_dev / torch.sqrt(torch.tensor(num_runs))

dat *= scale
Y *= scale
variance *= scale
confidence *= scale

print('output shape:',Y.shape)

# -----------------------------------------------------------------------------#
# Plotting:
# Combine data
dat = torch.from_numpy(dat)
all = torch.cat((dat, Y), dim=0)

# Scale down full data (global normalisation)
incident_max = -999999999
mention_max = -999999999

for i in range(all.shape[0]):
    for j in range(all.shape[1]):
        if 'WAR' in col[j] or 'Holiday' in col[j] or j in range(16, 32):
            continue

        if 'Mention' in col[j]:
            if all[i, j] > mention_max:
                mention_max = all[i, j]

        else:
            if all[i, j] > incident_max:
                incident_max=all[i, j]

all_n = torch.zeros(all.shape[0], all.shape[1])
confidence_n = torch.zeros(confidence.shape[0], confidence.shape[1])
u = 0

for i in range(all.shape[0]):
    for j in range(all.shape[1]):
            if 'Mention' in col[j]:
                all_n[i, j] = all[i, j] / mention_max
            else:
                all_n[i, j] = all[i, j] / incident_max
            
            if i >= all.shape[0] - 36:
                confidence_n[u, j] = \
                    confidence[u, j] * (all_n[i, j] / all[i, j])

    if i >= all.shape[0] - 36:
        u += 1

# Smoothing
smoothed_dat = torch.stack(exponential_smoothing(all_n, 0.1))
smoothed_confidence = torch.stack(exponential_smoothing(confidence_n, 0.1))
done = []

# Plot all forecasted pertinent technologies
for attack, solutions in graph.items():
    for s in solutions:
        if not s in done:
            done.append(s)
            plot_forecast(
                smoothed_dat[:-36, ],
                smoothed_dat[-36:, ],
                smoothed_confidence,
                s, index, col
            )
