
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as mt
import collections
import powerlaw
from scipy.optimize import curve_fit



def summary_statistics(Gs):
	
	sum = collections.defaultdict(list)
	
	for i in range(len(Gs)):
		sum['Nodes'].append(len(Gs[i].nodes()))
		sum['Edges'].append(len(Gs[i].edges()))
		sum['Density'].append(nx.density(Gs[i]))
	
	return pd.DataFrame(sum, index=Gs.index)
          
	
def hist_weight_dist(G,nbins = 30,dens = True,w_normalized = True,ylog=False):
        
	w_list = np.array([k['weight'] for u,v,k in G.edges(data=True)])
	
	if w_normalized:
		w_list = w_list/max(w_list)

	plt.hist(w_list,bins = nbins,density=dens,label = 'Weight distribution')

	if ylog:
		plt.yscale('log')
	
	plt.xlabel('Weight')
	plt.ylabel('freq')
	plt.legend()
	plt.show()
 
def get_weight_dist(G):
	
	sequence = [d['weight'] for u,m,d in G.edges(data=True)]

	Count = collections.Counter(sequence)

	seq, count = zip(*Count.items())

	# normalize counts
	freq = list(map(lambda x: x/sum(count),count))

	return seq,freq

	
def hist_degree_dist(G,nbins = 30,dens = True,ylog=False):
		
	plt.hist([d for n, d in G.degree()],bins = nbins,density=dens,label = 'Degree distribution')

	if ylog:
		plt.yscale('log')
	
	plt.xlabel('Degree')
	plt.ylabel('freq')
	plt.legend()
	plt.show()
	
def get_degree_dist(G):
	"""
	Return list of degree sores and their frequency
	"""
	
	sequence = sorted([d for n, d in G.degree()],reverse = False)  

	Count = collections.Counter(sequence)

	seq, count = zip(*Count.items())

	# normalize counts
	freq = list(map(lambda x: x/sum(count),count))

	return seq,freq
	
	
def binned_values(G,nbins=15):

	"""
	Return degree values for each bin and the respective frequency
	"""
	
	# Get a list of degrees for each node
	degree_list = list(dict(nx.degree(G)).values())

	# Get maximum and minimum degree
	kmin=min(degree_list)
	kmax=max(degree_list)

	start = max(np.log10(kmin),0)

	# Return numbers spaced evenly on a log scale
	# The sequence starts at base to the power of start and ends at base to the power of stop (included)
	logBins = np.logspace(start,np.log10(kmax),num=nbins,endpoint=True)
	  
	# smallest integer value greater than or equal to x
	for x in range(len(logBins)):
		logBins[x] = mt.ceil(logBins[x])
	   
	# a: is an array of values 
	# bins: it defines the bin edges, including the rightmost edge, allowing for non-uniform bin widths
	# density = True: the result is the value of the probability density function at the bin normalized
	  
	logBinDensity = np.histogram(a=degree_list, bins=logBins,density=True)[0]  

	# Delete the rightmost bin in order to get the same length as logBinDensity
	logBins = np.delete(logBins, -1)

	return logBins,logBinDensity
	
	
def hist_dc_dist(G,nbins = 30,dens = True,ylog=False):
		
	plt.hist(list(nx.degree_centrality(G).values()),bins = nbins,density=dens,label = 'degree_centrality distribution')

	if ylog:
		plt.yscale('log')
		
	plt.xlabel('Degree_centrality')
	plt.ylabel('freq')
	plt.show()
	plt.legend()
	
def get_dc_dist(G):
	"""
	Return list of degree_centrality scores and their frequency
	"""
	
	sequence = sorted([d for n, d in nx.degree_centrality(G)])  

	Count = collections.Counter(sequence)

	seq, count = zip(*Count.items())

	# normalize counts
	freq = list(map(lambda x: x/sum(count),count))

	return seq,freq


def hist_bc_dist(G,nbins = 30,dens = True,ylog=False):
		
	plt.hist(list(nx.betweenness_centrality(G).values()),bins = nbins,density=dens,label = 'Betweenness_centrality distribution')

	if ylog:
		plt.yscale('log')
	
	plt.xlabel('Betweenness_centrality')
	plt.ylabel('freq')
	plt.show()
	plt.legend()
	
def get_bc_dist(G):
	"""
	Return list of betweenness_centrality scores and their frequency
	"""
	
	sequence = sorted([d for n, d in nx.betweenness_centrality(G)])  

	Count = collections.Counter(sequence)

	seq, count = zip(*Count.items())

	# normalize counts
	freq = list(map(lambda x: x/sum(count),count))

	return seq,freq
	

def slice_network(G, T, copy=True):
	"""
	Remove all edges with weight < T from G or its copy. T must vary between 0 and 1
	"""
	F = G.copy() if copy else G
	F.remove_edges_from([(n1, n2) for n1, n2, w in F.edges(data=True) if  np.isnan(w['weight']) or w['weight'] < T ])
	return F
	
def slice_network_seq(Gs,T):
	
	Fs = []
	for i in range(len(Gs)):
		Fs.append(slice_network(Gs[i],T,copy = True))
		Fs[i].remove_nodes_from(list(nx.isolates(Fs[i])))
	
	return pd.Series(Fs,index = Gs.index)
	
def lin_reg(x,y,nbins=8,start_bin = 2,end_bin = 8):

	"""
	linear fit of log transformed degree distribution  with linregress scipy method 
	"""
	# Get slope and intercept
	lm = linregress(x[start_bin:end_bin+1],y[start_bin:end_bin+1])
	
	return lm
	
def lin_reg_plot(G,nrows = 3,ncols=3,size = (15,15),nbins = 8,start_bin =2, start_cutoff = 0.1,end_cutoff = 0.3):
	
	"""
	Plot nrows*ncols log-transformed degree distributions for different weight thresholds
	"""
	
	cut_off = np.linspace(start_cutoff,end_cutoff,nrows*ncols)

	_, plot = plt.subplots(nrows=nrows,ncols=ncols,figsize = size)
	subplots = plot.reshape(1, nrows*ncols)[0] 

	for i,subplot in enumerate(subplots):
		
		F = slice_network(G,cut_off[i])  
		
		#print('Graph number: ',i+1)
		print()
		print('Weight: ',cut_off[i])
		summary_statistics(F)

		
		F.remove_nodes_from(list(nx.isolates(F)))	
		print('After slicing network: ')
		summary_statistics(F)

		d,f = get_degree_dist(F)
		d,f = np.log(d),np.log(f)
		
		bins,dens = binned_values(F,nbins)
		x,y = np.log(bins),np.log(dens)
		
		lm = lin_reg(x,y,start_bin = start_bin,end_bin = nbins)
		print('p-value: ',lm.pvalue)
		print('r correlation coefficient: ',lm.rvalue)
		print('#########################################')
		print()
		
		subplot.plot(x,x*lm.slope + lm.intercept,'g--',label= 'linear fit: slope:{0}, sd:{1} '.format(np.round(lm.slope,3),np.round(lm.stderr,3)))
		subplot.scatter(x,y,s=10,color='b',label = 'binned data')
		subplot.scatter(d,f,s=10,c='r',alpha=0.3,label= 'data')
		subplot.set_ylim((min(f)-0.5,max(f)+0.5))
		subplot.set_xlim((min(d)-0.5,max(d)+0.5))
		
		
		subplot.xlabel('log - Degree')
		subplot.legend('log - freq')
 
	plot.title('Degree Distributions')
 
def func(x, a, b):
    return a*np.power(x,-b)
	

def curve_fit_plot(G,nrows = 3,ncols=3,size = (15,15),nbins = 12, start_cutoff = 0.1,end_cutoff = 0.3):
	
	"""
	Plot in log-log scale nrows*ncols degree distributions fitted with curve_fit function
	
	The function fitted is: a*np.power(x,-b)
	
	a,b reported on each plot
	
	Return dataframe with statistical summary
	"""
	
	cut_off = np.linspace(start_cutoff,end_cutoff,nrows*ncols)

	_, plot = plt.subplots(nrows=nrows,ncols=ncols,figsize = size)
	subplots = plot.reshape(1, nrows*ncols)[0] 
	
	stats = collections.defaultdict(list)
	fit = collections.defaultdict(list)
	
	for i,subplot in enumerate(subplots):
		
		F = slice_network(G,cut_off[i]) 
		
		stats['nodes'].append(len(F.nodes()))
		stats['edges'].append(len(F.edges()))
		stats['density'].append(nx.density(F))
		stats['isolates'].append(nx.number_of_isolates(F))
		
		F.remove_nodes_from(list(nx.isolates(F)))
		
		stats['nodes'].append(len(F.nodes()))
		stats['edges'].append(len(F.edges()))
		stats['density'].append(nx.density(F))
		stats['isolates'].append(nx.number_of_isolates(F))
		
		data = list(dict(nx.degree(F)).values())
		d,f = get_degree_dist(F)
		
		bins,dens = binned_values(F,nbins)
		
		powerlaw.plot_pdf(data,linestyle = '-.',c = 'b', label = 'fit log-bins',alpha = 0.6,ax = subplot)
		subplot.scatter(bins,dens,s=10,c='b',label = 'binned data')
		subplot.scatter(d,f,s=10,c='r',alpha=0.3, label ='data')
		subplot.set_xscale('log')
		subplot.set_yscale('log')
		subplot.title.set_text('Weight - slice: {}'.format(round(cut_off[i],2)))
		subplot.set_ylim((min(f)-10e-4,max(f)+10e-1))
		subplot.set_xlim((min(d),max(d)+200))
		subplot.legend()
		subplot.set_xlabel('Degree')
		subplot.set_ylabel('freq')
	
	
	ir = ['F','T']
	cf = np.sort(np.append(cut_off,cut_off))
	
	l = [(c,ir[i%2]) for i,c in enumerate(cf)]	
	
	df = pd.DataFrame(stats,index = pd.MultiIndex.from_tuples(l,names=['Weight','isolates removed']))
	
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
 
	return df
	


