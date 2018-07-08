from math import sqrt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse.csr import csr_matrix
from scipy.sparse.csc import csc_matrix
import pickle

import scipy.sparse as sp
from scipy.sparse.linalg import svds

import time
import datetime
import random
import networkx as nx
import networkx.algorithms as bipartite

import itertools
from itertools import combinations

from collections import Counter
from collections import defaultdict


	
def set_user_attr(G):

	"""
	Open genre_1_dict,genre_2_dict in the same folder and add them to user node attributes.
	These attributes will not be considered when computing the similarities between users, but may help visualize the graph on Gephi.

	"""
	
	
	f = open("attributes/best_genre_user.pkl","rb")
	genre_dict = pickle.load(f)
	f.close()

	f = open("attributes/partition_user.pkl","rb")
	partition_dict = pickle.load(f)
	f.close()
	
	genre_1_dict = {u:k[0] for u,k in genre_dict.items()}
	genre_2_dict = {u:k[1] for u,k in genre_dict.items()}
	genre_3_dict = {u:k[2] for u,k in genre_dict.items()}
	
	
	nx.set_node_attributes(G,genre_1_dict,'genre1')
	nx.set_node_attributes(G,genre_2_dict,'genre2')
	nx.set_node_attributes(G,genre_3_dict,'genre3')
	nx.set_node_attributes(G,partition_dict,'partition')
	

	return G


def user_projection_pearson(G,users):
	
	"""
	Perform user projection of a bipartite graph user|movie using cosine metric and scipy_sparse matrix representation
	"""

	mat = nx.bipartite.biadjacency_matrix(G,users,weight = 'rating')
	user_sim = csr_matrix(1 - pairwise_distances(mat.toarray(), metric='correlation') - np.eye(len(users)),dtype=np.float32)

	Gu = nx.from_scipy_sparse_matrix(user_sim)
	
	Gu = nx.relabel_nodes(Gu,{i:n for i,n in enumerate(users)})
	Gu = set_user_attr(Gu)

	return Gu


# Recommendation using Collaborative filter
def graph_reccomendation_col(Gs,Gs_user,df_movie,users,movies,start_date = datetime.datetime(2002,1,1),threshold = 0.12,alpha=0.3,min_new_items=10):
	
	# find Gs and Gs_user at start_date
	id_min = max(np.where(Gs.index <= start_date)[0])
	
	# Repeat copy for storing RS - Driven network sequence
	Ns = []
	Ns_user = []
	
	for i in range(len(Gs)):
		Ns.append(Gs[i].copy(as_view=False))
		Ns_user.append(Gs_user[i].copy(as_view=False))
	
	# Step between 2 successive snapshots
	window = 1
	
	# Store statistics 
	stats = defaultdict(list)
	start_time = time.clock()
	
	
	for i in np.arange(id_min,len(Ns)-1):
		
		df = df_movie[df_movie['year'] <= Gs.index[i+1].year]
	
		# Find new user - movie links added in next snapshot
		target_u = [ u for u in Ns_user[i].nodes() if u in Ns[i+window].nodes()]
		target_u = list(filter(lambda x: len(set(Ns[i+window].neighbors(x)).difference(Ns[i].neighbors(x))) > 0,target_u))
		
		neigh_u = [u for u in Ns_user[i].nodes() if u in Ns[i+window].nodes()]	
		
		#print('1',time.clock() - start_time, "seconds")
		
		# Dictionary of number of movies watched in next time frame 
		len_new_items = {u:len(set(Ns[i+window].neighbors(u)).difference(Ns[i].neighbors(u))) for u in target_u}
		
		# Dictionary of list of titles watched by the user in //
		list_new_items = {u: list(set(Ns[i+window].neighbors(u)).difference(Ns[i].neighbors(u))) for u in target_u}
		
		# Dictionary of list of titles watched by users over the last time frame
		list_last_items = {u: list(set(Ns[i+window].neighbors(u)).difference(Ns[i-window].neighbors(u))) for u in neigh_u}
		
		# Dictionary of list of watched by the user until the current time
		list_all_items = {u: list(Ns[i].neighbors(u)) for u in target_u}
		
		# Dictionary of user neighbors in the projected graph	
		user_neigh = {u: list(filter(lambda x: x in neigh_u,list(Ns_user[i].neighbors(u)))) for u in target_u}
			
		#print('2',time.clock() - start_time, "seconds")
		
		# Store precision 
		prec = []	
		movie_nr = []
		movie_cat = []
		movie_r = []
		us = 0
				
		for u in target_u:
			
			l_items = defaultdict(list)
						
			for v in user_neigh[u]:
				for k in list_last_items[v]:
					if k not in list_all_items[u]:
						try:
							if  Ns[i+window][u][k]['date'] > Ns[i+window][v][k]['date']:
								l_items[k].append(Ns_user[i][u][v]['weight'])
						except:
							l_items[k].append(Ns_user[i][u][v]['weight'])
			
			
			l_items = list(pd.Series(dict(l_items)).apply(sum).sort_values(ascending=False).index)
			
			#print('3:',u,time.clock() - start_time, "seconds")
			
			# number of random items
			#r = int(alpha*len_new_items[u])
			# number of non-random items 
			#nr = len_new_items[u] - r
			
			nr = np.random.binomial(len_new_items[u],alpha)
			r = len_new_items[u] - nr 
							
			# Take nr items from the recommended list
			l_nr = l_items[:nr]
			
			movie_nr.append(len(l_nr))
			
			#print('len all items u',len(list_all_items[u]))
			#print()
			#print('recommended list',l_nr)
			
			# Take at random a movie that:
			# 1) it is not included in the catalog of the user
			# 2) it is not included in l_nr
			# 3) Year of release is before the current time
			# 4) Genre is one of the most preferred by user
			
			mov = df[(df.genres.str.contains(Ns_user[i].node[u]['genre1']))| (df.genres.str.contains(Ns_user[i].node[u]['genre2']))|(df.genres.str.contains(Ns_user[i].node[u]['genre3']))]
			mov = list(set(df['movieId']).difference(l_items).difference(list_all_items[u]))
			
			movie_cat.append(len(mov))
			
			#set random seed
			random.seed(np.float(time.time()))	
			
			l_r = np.random.choice(mov,size = r,replace=False)
			
			movie_r.append(len(l_r))
			l_fin = list(set(l_nr).union(l_r))
				
			# Filter users who will link to at least min_new_items in next snapshot
				
			if len_new_items[u] >= min_new_items:
				prec.append(np.float(len(set(l_fin).intersection(list_new_items[u]))/len_new_items[u]))
				us +=1
				
			# replace list of new items u with the new recommended list
			weights = np.random.choice(a = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5],size=int(len_new_items[u]), replace=True, p=[0.02,0.03,0.02,0.08,0.05,0.20,0.10,0.3,0.05,0.15])
			Ns[i+window].remove_edges_from(e for e in set(Ns[i+window].edges(u)).difference(Ns[i].edges(u)))
			Ns[i+window].add_weighted_edges_from([(u,l_fin[i],weights[i]) for i in range(len(l_fin))],'rating')
				
		Ns_user[i+window] =  user_projection_pearson(Ns[i+window],users)
		Ns_user[i+window] = slice_network(Ns_user[i+window],T = threshold,copy = False)
		Ns_user[i+window].remove_nodes_from(list(nx.isolates(Ns_user[i+window])))
		Ns_user[i+window] = set_user_attr(Ns_user[i+window])
		
		
		#movie_nr = list(itertools.chain.from_iterable(movie_nr))
		#movie_nr = dict(Counter(movie_nr))
		
		#print('6',time.clock() - start_time, "seconds")
		
		stats['avg_prec'].append(np.mean(prec))
		stats['std_prec'].append(np.std(prec))
		stats['cov_users_prec'].append(us)
		
		stats['users'].append(target_u)
		
		stats['movie_nr'].append(np.mean(movie_nr))
		stats['movie_r'].append(np.mean(movie_r))
		stats['movie_cat'].append(np.mean(movie_cat))
		
	return pd.Series(Ns,index=Gs.index), pd.Series(Ns_user,index=Gs.index),pd.DataFrame(stats,index = Gs.index[id_min+1:])
				
		
# Recommendation using Content - based filter
def get_nodes_and_nbrs(G, nodes_of_interest):
    """
    Returns a subgraph of the graph `G` with only the `nodes_of_interest` and their neighbors.
    """
    nodes_to_draw = set()
    
    # Iterate over the nodes of interest
    for i,j in itertools.combinations(nodes_of_interest,2):
        
        s = set(nx.astar_path(G,i,j))
        # Append the nodes of interest to nodes_to_draw
        nodes_to_draw = nodes_to_draw | s
    
    return G.subgraph(nodes_to_draw).copy()
		
def top_dc_connected(G,n=5):
	
	"""
	Get the nodes having n unique highest degree centrality scores
	"""
	top_dcs = sorted(set(nx.degree_centrality(G).values()), reverse=True)[0:int(n)]

	# Create list of nodes that have the top n highest overall degree centralities
	top_connected = []
	for n, dc in nx.degree_centrality(G).items():
		if dc in top_dcs:
			top_connected.append(n)

	return top_connected


				
def slice_network(G, T, copy=True):
	"""
	Remove all edges with weight < T from G or its copy. T must vary between 0 and 1
	"""
	F = G.copy() if copy else G
	F.remove_edges_from([(n1, n2) for n1, n2, w in F.edges(data=True) if np.isnan(w['weight']) or w['weight'] < T])
	return F
				
			
def triadic_closure(Gs_user,window = 1,start_date = datetime.datetime(1997,1,1),max_k = 12):
	
	"""
	Probability of forming a link bet 2 users having k friends in common
	
	It's not said that the set of users considered must form a link in next step
	"""
	
	# Find Gs and Gs_user representing networks at start_date
	id = max(np.where(Gs_user.index < start_date)[0])
	
	Fs_user = Gs_user[id:].copy()
	
	df = defaultdict(list)
	
	
	start_time = time.clock()
	
	for i in range(len(Fs_user) - window):
		
		new_edges = set(Fs_user[i+window].edges()).difference(Fs_user[i].edges())
		fraction = []
		
		#print(1,i,time.clock()-start_time)
		
		for k in range(max_k+1):
			
			# List of nodes having currently k or more neighbors
			users_k  = {u:list(Fs_user[i].neighbors(u)) for u in Fs_user[i].nodes() if len(list(Fs_user[i].neighbors(u))) >= k} 
			#if u in g2.nodes()}
			
			#print(2,i,k,time.clock()-start_time)
			# Get the number of users having k friends in common in the first snapshot, but not directly connected by an edge
			triadic_links = [(u,v) for u,v in combinations(users_k.keys(),2) if not Fs_user[i].has_edge(u,v) and  len(set(users_k[u]).intersection(users_k[v])) == k ]		
			#print(3,i,k,time.clock()-start_time)
			
			# Get fraction of edges that have actually formed in the 2nd snapshot
			new_links = [(u,v) for u,v in triadic_links if (u,v) in new_edges or (v,u) in new_edges]
			
			try:
				fraction.append(len(new_links)/len(triadic_links))
			except:
				continue

		df[i] = fraction
	
	return pd.DataFrame(df)

def focal_closure(Gs,Gs_user_sliced,start_date = datetime.datetime(1997,1,1),window=1,max_k = 10):
	
	"""
	Probability of forming a link bet 2 users having k preferred movies in common
	
	It's not said that the set of users considered must necessarily form a link in next step
	
	
	"""
	# Find Gs and Gs_user representing networks at start_date
	id = max(np.where(Gs.index < start_date)[0])
	
	# Copy of Gs and Gs_user_sliced
	Fs = Gs[id:].copy()
	Fs_user = Gs_user_sliced[id:].copy()
	
	frac = defaultdict(list)
	
	for i in range(len(Fs_user) - window):
		
		g1 = Fs[i]
		
		f1 = Fs_user[i]
		f2 = Fs_user[i+window]
		
		# Compute graph difference here
		f = f2.copy()
		f.remove_edges_from(e for e in f2.edges() if e in f1.edges())
		f.remove_nodes_from(list(nx.isolates(f)))
		
		fraction = []
		
		for k in range(1,max_k+1):
		
			# List of nodes having currently at least k preferred movies			
			users_k  = {u:list(g1.neighbors(u)) for u in f1.nodes() if len(list(g1.neighbors(u))) >= k} 
			#if u in f2.nodes()}
			# u in g1.nodes() and 
			
			# Dictionary of list of friends linked by the user in next step
			list_new_friends = {u: list(f.neighbors(u)) for u in f.nodes()}
			
			# Get the number of users having k preferred movies in common in the first snapshot, but not directly connected by an edge
			triadic_links = [(u,v) for u,v in combinations(users_k.keys(),2) if not f1.has_edge(u,v) and len(set(users_k[u]).intersection(users_k[v])) == k ]
			
			# Get fraction of edges that have actually formed in the 2nd snapshot
			new_links = [(u,v) for u,v in triadic_links if u in f.nodes() and v in list_new_friends[u]]
			
			try:
				fraction.append(len(new_links)/len(triadic_links))
			except: 
				fraction.append(0)
				
		frac[i] = fraction
	
	frac = pd.DataFrame(frac)
	
	return frac	

def membership_closure(Gs,Gs_user_sliced,start_date = datetime.datetime(1997,1,1),window=1,max_k = 10):
	
	"""
	Probability of forming a link bet user and movie if k friends have watched it 
	
	It's not said that the set of users considered must necessarily create a link in next step
	
	"""
	
	# Find Gs and Gs_user representing networks at start_date
	id = max(np.where(Gs_user_sliced.index < start_date)[0])
	
	#Copy of Gs and Gs_user_sliced
	Fs = Gs[id:].copy()
	Fs_user = Gs_user_sliced[id:].copy()
	
	frac = defaultdict(list)
	
	for i in range(len(Fs_user) - window):
		
		g1 = Fs[i]
		g2 = Fs[i+window]
		g = g2.copy()
		g.remove_edges_from(e for e in g2.edges() if e in g1.edges())
		g.remove_nodes_from(list(nx.isolates(g)))
		
		f1 = Fs_user[i]
		
		fraction = []
			
		for k in range(max_k+1):
		
			# List of nodes having currently at least k friends	(It's not said that the user must necessarily create a link in next step)		
			users_k  = {u:list(f1.neighbors(u)) for u in f1.nodes() if len(list(f1.neighbors(u))) >= k if u}
			#u in g.nodes() not required
			
			# Dictionary of list of items already linked by each user
			list_current_items = {u: list(g1.neighbors(u)) for u in f1.nodes()}
			
			# Dictionary of list of items linked by the user in next step
			list_new_items = {u: list(g.neighbors(u)) for u in users_k if u in g.nodes()}
			
			nl = 0
			tl = 0
			
			for u in users_k.keys():
			
				l_it = [list_current_items[k] for k in users_k[u]]
				l_items = list(itertools.chain.from_iterable(l_it))
			
				dict_it = dict(Counter(l_items))
				
				# filter the list so that the items have not already been watched by the user
				l_items = list(filter(lambda x: x not in list_current_items[u],l_items))
				
				l_items = [v for v,c in dict_it.items() if c >= k]
			
				# Get fraction of edges that have actually formed in the 2nd snapshot		
				try:
					new_links = set(list_new_items[u]).intersection(l_items)
				except:
					continue

				nl =+ len(new_links)
				tl =+ len(l_items)
			
			if tl >0:
				fraction.append(nl/tl)
			else:
				fraction.append(0)
				
				
		# Take the mean on horizontal axis
		frac[i] = fraction
	
	frac = pd.DataFrame(frac)
	
	return frac
