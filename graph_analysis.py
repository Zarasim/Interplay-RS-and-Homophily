import pandas as pd
import numpy as np

import networkx as nx
from networkx.algorithms import bipartite
import datetime
import pickle
import collections


from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist
from scipy.sparse.csr import csr_matrix



genres = sorted(['Adventure','Animation','Children','Drama','Documentary','Horror','Musical','Thriller','Action',
              'Sci-Fi','War','Romance','Comedy','Crime','Fantasy','Western','Mystery','IMAX','Film-Noir'])

def create_bipartite_graph(df,rating = True,rating_cutoff = 0.0):
	
	"""
	Create a bipartite Graph given a DataFrame with column names: 'userId', 'movieId', 'rating', 'date'
	
	type(UserId): int | bipartite node attribute: 'user'
	type(MovieId): str | bipartite node attribute: 'movie'
	type(rating): float
	type(date): datetime.datetime

	Store rating as optional edge attribute for projections performed by distance similarities (Cosine/Pearson) 
	"""
	# Create graph with edge_attr rating and date
	df_red = df[df.rating >= rating_cutoff]
	
	users = sorted(df_red.userId.unique())
	movies = sorted(df_red.movieId.unique())
	
	if rating:
		G = nx.from_pandas_edgelist(df_red, source= 'userId',target='movieId',edge_attr=['rating','date'])
	else:
		G = nx.from_pandas_edgelist(df_red, source= 'userId',target='movieId',edge_attr=['date'])
	
	# Set node_attr
	nx.set_node_attributes(G,{n:'user' for n in G.nodes() if type(n) == int},'bipartite')
	nx.set_node_attributes(G,{n:'movie' for n in G.nodes() if type(n) == str},'bipartite')

	# Assert the graph is bipartite
	assert nx.is_bipartite(G)

	return G,users,movies

# set attributes
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
	#nx.set_node_attributes(G,partition_dict,'partition')
	
	return G

def set_movie_attr(G):
    
	"""
	Open genre_mov_dict,title_dict,year_dict in the same folder and add them as movie node attributes.
	"""

	f = open("attributes/genre_mov_dict.pkl","rb")
	gen_mov_dict = pickle.load(f)
	f.close()

	f = open("attributes/title_dict.pkl","rb")
	title_dict = pickle.load(f)
	f.close()

	f = open("attributes/year_dict.pkl","rb")
	year_dict = pickle.load(f)
	f.close()

	nx.set_node_attributes(G,gen_mov_dict,'genre')
	nx.set_node_attributes(G,year_dict,'year')
	nx.set_node_attributes(G,title_dict,'title')
	
	return G

def graph_seq(G,start_date = datetime.datetime(1996,1,1),end_date = datetime.datetime(2017,1,1),tw_days = 365):    
    
	""" 
	Build graph sequence starting from 1 Jan of 1996 until 1 Jan 2016 with sliding time window of tw_days. 
	
	Return the sequence of graphs Gs in a Series
	"""
	# Initialize an empty list: Gs
	Gs = []

	date = start_date
	dates = []
	
	while date < end_date:

		dates.append(date)
		# Instantiate a new undirected graph: F
		F = nx.Graph()
		F.add_nodes_from(G.nodes(data=True))
		
		# To make comparison set equal number of nodes
		
		F.add_edges_from([(u,m,k) for (u,m,k) in G.edges(data=True) if datetime.datetime.strptime(k['date'],'%Y-%m-%d') >= start_date 
							  and datetime.datetime.strptime(k['date'],'%Y-%m-%d') < date + datetime.timedelta(tw_days)])

		# Append G to the list of graphs
		Gs.append(F)
	
		# increment year by tw
		date += datetime.timedelta(tw_days)
				
				
	return pd.Series(Gs,index = dates)
	
	
	
def added_edges(Gs,window = 1):
	
	"""
	Return fraction of edges added, removed and their fractional_change over time
	
	"""
	changes = []

	for i in range(len(Gs) - window):
		g1 = Gs[i]
		g2 = Gs[i + window]
		
		g = g2.copy()
		g.remove_edges_from(e for e in g2.edges() if e in g1.edges())

	
		changes.append(len(g.edges()))
		
	return pd.Series(changes,index = Gs.index.values[1:])
	
def removed_edges(Gs,window = 1):
	
	"""
	Return fraction of edges added, removed and their fractional_change over time
	
	"""
	changes = []

	for i in range(len(Gs) - window):
		g1 = Gs[i]
		g2 = Gs[i + window]
		
		g = g1.copy()
		g.remove_edges_from(e for e in g1.edges() if e in g2.edges())

	
		changes.append(len(g.edges()))
		
	return pd.Series(changes,index = Gs.index.values[1:])
	
def get_density(Gs):
	
	"""
	Return density change over time
	
	"""
	d = []
	for i in range(len(Gs)):
		d.append(nx.density(Gs[i]))
    
	return pd.Series(d,index = Gs.index)	

def edge_seq(Gs):

	nedges = []
	for i in range(len(Gs)):
		nedges.append(len(Gs[i].edges()))
		
	return pd.Series(nedges,index = Gs.index)


# Projections cosine similarity 
	
def user_projection_cosine_seq(Gs):
	
	Gs_user = []
	
	for i in range(len(Gs)):
		
		G_user = user_projection_cosine(Gs[i])
		Gs_user.append(G_user)

	return pd.Series(Gs_user,index = Gs.index)
	
def movie_projection_cosine_seq(Gs):
	
	Gs_movie = []
	
	for i in range(len(Gs)):
		
		G_movie = movie_projection_cosine(Gs[i])
		Gs_movie.append(G_movie)

	return pd.Series(Gs_movie,index = Gs.index)
	
	
	
def user_projection_cosine(G):
	
	"""
	Perform user projection of a bipartite graph user|movie using cosine metric and scipy_sparse matrix representation
	"""
	
	users = [u for u,k in G.nodes(data=True) if k['bipartite'] == 'user']
	items = [u for u,k in G.nodes(data=True) if k['bipartite'] == 'movie']
	
	mat = nx.bipartite.biadjacency_matrix(G,users,items,weight = 'rating')
	user_sim = csr_matrix(1 - pairwise_distances(mat, metric='cosine') - np.eye(len(users)),dtype=np.float32)

	Gu = nx.from_scipy_sparse_matrix(user_sim)
	Gu = nx.relabel_nodes(Gu,{i:n for i,n in enumerate(users)})
	Gu = set_user_attr(Gu)

	return Gu

def movie_projection_cosine(G):
	
	"""
	Perform movie projection of a bipartite graph user|movie using cosine metric and scipy_sparse matrix representation
	"""
	
	users = [u for u,k in G.nodes(data=True) if k['bipartite'] == 'user']
	items = [u for u,k in G.nodes(data=True) if k['bipartite'] == 'movie']
	
	mat = nx.bipartite.biadjacency_matrix(G,users,movies,weight = 'rating')
	movie_sim = csr_matrix(1 - pairwise_distances(mat.T, metric='cosine') - np.eye(len(movies)),dtype=np.float32)

	Gm = nx.from_scipy_sparse_matrix(movie_sim)
	Gm = nx.relabel_nodes(Gm,{i:n for i,n in enumerate(movies)})
	
	Gm = set_movie_attr(Gm)

	return Gm


# Euclidian projection
def user_projection_euclidian(G):
	
	"""
	Perform user projection of a bipartite graph user|movie using cosine metric and scipy_sparse matrix representation
	"""
	
	users = [u for u,k in G.nodes(data=True) if k['bipartite'] == 'user']
	items = [u for u,k in G.nodes(data=True) if k['bipartite'] == 'movie']
	
	mat = nx.bipartite.biadjacency_matrix(G,users,items,weight = 'rating')
	a = pairwise_distances(mat, metric='l2')
	user_sim = csr_matrix(1 - (a - a.min(axis=1))/(a.max(axis=1) - a.min(axis=1)) - np.eye(len(users)),dtype=np.float32)
	
	Gu = nx.from_scipy_sparse_matrix(user_sim)
	
	Gu = nx.relabel_nodes(Gu,{i:n for i,n in enumerate(users)})
	Gu = set_user_attr(Gu)

	return Gu

	
# Pearson projection

def user_projection_pearson_seq(Gs,users):
	
	Gs_user = []
	for i in range(len(Gs)):
		Gu = user_projection_pearson(Gs[i],users)
		Gs_user.append(Gu)
	
	return pd.Series(Gs_user,index=Gs.index)
	

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



# Projection jaccard similarity

def user_projection_jac_seq(Gs,users):
	Gs_user = []
	for i in range(len(Gs)):
		Gu = user_projection_jac(Gs[i],users)
		Gs_user.append(Gu)
	
	return pd.Series(Gs_user,index=Gs.index)
	
def user_projection_jac(G,users):

		"""
		Perform user projection of a bipartite graph user|movie using overlapping metric
		"""
		
		Gu = nx.bipartite.overlap_weighted_projected_graph(G,nodes=users)
		Gu = set_user_attr(Gu)
		
		return Gu
		

def movie_projection_jac_seq(Gs):
	
	Gs_movie = []
	for i in range(len(Gs)):
		Gm = movie_projection_jac(Gs[i])
		Gs_movie.append(Gm)
	
	return pd.Series(Gs_movie,index=Gs.index)
	
def movie_projection_jac(G):


		"""
		Perform user projection of a bipartite graph user|movie using overlapping metric
		"""
		
		movies = [u for u,k in G.nodes(data=True) if k['bipartite'] == 'movie']
		Gm = nx.bipartite.overlap_weighted_projected_graph(G,nodes=movies)
		Gm = set_movie_attr(Gm)
		
		return Gm
		

# Similarity in User Genre
	
def user_genre_similarity(G):

	"""
	Get a Series with similarity for each genre from dict in which keys are genres and values are numeric assortativity coefficients
	"""
	Ser = pd.Series({genres[j]:nx.numeric_assortativity_coefficient(G,genres[j]) for j in range(len(genres))})

	return Ser

def graph_seq_similarity(Gs):
	"""
	Get DataFrame indexed by starting and ending date with genre as column names
	Series must be transposed and concatenated by row 
	"""
	Simframe = []

	for i in range(len(Gs)):
		
		# The keys are the name of the columns
		Simframe.append(pd.DataFrame({genres[j]:nx.numeric_assortativity_coefficient(Gs[i],genres[j]) for j in range(len(genres))},index = [Gs.index.values[i]])) 	
		
	return pd.concat(Simframe)
		
# Centrality measures
		
def set_dc_attr(G):
	"""
	Set degree centrality attribute for each node
	"""
	nx.set_node_attributes(G,nx.degree_centrality(G),'dc')
	
def dc_seq(Gs):

	"""
	Create list of dictionaries of degree_centrality score at each time step
	"""

	# Create a list of degree centrality scores
	seq = defaultdict(list)
	for i,G in enumerate(Gs):
		cent = nx.degree_centrality(G)
		seq[i].append(list(cent.values()))

	return seq

	
def set_bc_attr(G):
	"""
	Set betweenness centrality attribute for each node
	"""
	nx.set_node_attributes(G,nx.betweenness_centrality(G),'bc')	

def bc_seq(Gs):

	"""
	Create list of dictionaries of betweenness_centrality score at each time step
	"""

	# Create a list of degree centrality scores
	seq = defaultdict(list)
	for G in Gs:
		cent = nx.betweenness_centrality(G)
		seq[i].append(cent)

	return seq
	
def centrality_measures(G):
	"""
	Return dict of dict where each key is a type of centrality: dgr,clo,bet,har,eig,pgr,hits
	"""
	
	cm = defaultdict(dict)

	cm[dgr] = nx.degree_centrality(G)
	cm[clo] = nx.closeness_centrality(G)
	cm[bet] = nx.betweenness_centrality(G) 
	cm[har] = nx.harmonic_centrality(G)
	cm[eig] = nx.eigenvector_centrality(G)

	return cm  

def centrality_measures_seq(Gs):
	"""
	Return list of centrality_measures for each graph in the sequence
	"""
	cents = []
	for G in Gs:
		cm = centrality_measures(G)
		cents.append(cm)

	return cents


# sequence of top centrality nodes
def top_dc__seq(Gs,top_dc_connected,dc_seq):

	"""
	Compute the dc score over time of the top dc connected nodes
	Return a dict in which the keys are top dc nodes and the values are a list of connectivity scores over time
	"""

	connectivity = defaultdict(list)
	for i,n in enumerate(top_dc_connected):
		connectivity[n].append(dc_seq[i][n])

	return connectivity

def top_bc__seq(Gs,top_bc_connected,bc_seq):

	"""
	Compute the bc score over time of the top bc connected nodes
	Return a dict in which the keys are top bc nodes and the values are a list of connectivity scores over time
	"""

	connectivity = defaultdict(list)
	for i,n in enumerate(top_bc_connected):
		connectivity[n].append(bc_seq[i][n])

	return connectivity


	
def subgraph_max_clique(G):

	"""
	Return the largest maximal clique with adjacent neighbors
	"""

	#Identify the largest maximal clique: largest_max_clique
	largest_max_clique = set(sorted(nx.find_cliques(G), key=lambda x: len(x))[-1])


	# Create a subgraph from the largest_max_clique: G_lmc
	G_lmc = G.subgraph(largest_max_clique)


	# Go out 1 degree of separation
	for node in G_lmc.nodes():
		G_lmc.add_nodes_from(G.neighbors(node))
		G_lmc.add_edges_from(zip([node]*len(G.neighbors(node)), G.neighbors(node)))

	# Record each node's degree centrality score
	set_dc_attr(G_lmc)
	set_bc_attr(G_lmc)

	return G_lmc

def get_gcc(G):

	comp_gen = nx.connected_components(G)
	# Find Greatest connected component  
	gcc = sorted(comp_gen, key = len, reverse = True)[0]
	 
	print('Size of gcc: ',len(gcc))

	return gcc