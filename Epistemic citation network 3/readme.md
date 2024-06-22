## Combined Results

Function to show Degree Distribution for PP,PW,WP,WW models side by side


```python
def degree_distribution_combined(G1,G2,G3,G4):
    """
    Shows the sorted degree distribution for 4 given graphs
 
    Args:
        G1 (graph): Network of authors
        G2 (graph): Network of authors
        G3 (graph): Network of authors
        G4 (graph): Network of authors
                
    Returns:
        float: 1D array of size equal to number of authors
    """
    
    fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(20,5))

    ax1.plot(np.sort(list_degrees(G1)))

    ax1.set_title('WP')
    ax1.set_xlabel('Authors')
    ax1.set_ylabel('# Authorships')
    ax1.grid(True)

    ax2.plot(np.sort(list_degrees(G2)))
    ax2.set_title('PP')
    ax2.set_xlabel('Authors')
    ax2.set_ylabel('# Authorships')
    ax2.grid(True)
    
    ax3.plot(np.sort(list_degrees(G3)))
    ax3.set_title('PW')
    ax3.set_xlabel('Authors')
    ax3.set_ylabel('# Authorships')
    ax3.grid(True)
    
    ax4.plot(np.sort(list_degrees(G4)))
    ax4.set_title('WW')
    ax4.set_xlabel('Authors')
    ax4.set_ylabel('# Authorships')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()    
```

Function to show heatmap of Epistemic Coverage for PP and WP models side by side


```python
def heatmap_combined(ec1,ec2,ec3,ec4):
    """
    Plots the heat map of epsitemic coverages for 4 given graphs
 
    Args:
        ec1 (float): epistemic coverage for a given graph
        ec2 (float): epistemic coverage for a given graph
        ec3 (float): epistemic coverage for a given graph
        ec4 (float): epistemic coverage for a given graph
                
    Returns:
        float: 2D array of size equal to number of authors
    """

    
    x1=np.zeros(len(ec1))
    y1=np.zeros(len(ec1))
    x2=np.zeros(len(ec2))
    y2=np.zeros(len(ec2))
    x3=np.zeros(len(ec3))
    y3=np.zeros(len(ec3))
    x4=np.zeros(len(ec4))
    y4=np.zeros(len(ec4))

    for i in range(len(ec1)):
        x1[i]=ec1[i][0]
        y1[i]=ec1[i][1]

    for i in range(len(ec2)):
        x2[i]=ec2[i][0]
        y2[i]=ec2[i][1]
        
    for i in range(len(ec3)):
        x3[i]=ec3[i][0]
        y3[i]=ec3[i][1]

    for i in range(len(ec4)):
        x4[i]=ec4[i][0]
        y4[i]=ec4[i][1]

    fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(20,5))

    
    h = ax1.hist2d(x1, y1)
    fig.colorbar(h[3], ax=ax1)
    ax1.set_title('WP')
    ax1.grid(True)

    h = ax2.hist2d(x2, y2)
    fig.colorbar(h[3], ax=ax2)
    ax2.set_title('PP')
    ax2.grid(True)
    
    h = ax3.hist2d(x3, y3)
    fig.colorbar(h[3], ax=ax3)
    ax3.set_title('PW')
    ax3.grid(True)
    
    h = ax4.hist2d(x4, y4)
    fig.colorbar(h[3], ax=ax4)
    ax4.set_title('WW')
    ax4.grid(True)
```

Function to show Network on heatmap for PP and WP models side by side


```python
def network_on_heatmap(G1,G2,G3,G4,ec1,ec2,ec3,ec4):
     """
    Plots the heat map of epsitemic coverages for 4 given graphs in epistemic space
 
    Args:
        G1 (graph): network of authors
        G2 (graph): network of authors
        G3 (graph): network of authors
        G4 (graph): network of authors
        ec1 (float): epistemic coverage for a given graph
        ec2 (float): epistemic coverage for a given graph
        ec3 (float): epistemic coverage for a given graph
        ec4 (float): epistemic coverage for a given graph
        
                
    Returns:
        float: 2D array of size equal to number of authors
    """

    x1=np.zeros(len(ec1))
    y1=np.zeros(len(ec1))
    for i in range(len(ec1)):
        x1[i]=ec1[i][0]
        y1[i]=ec1[i][1]
    coordinates1 =ec1
    pos1 = {i: (coordinates1[i][0], coordinates1[i][1]) for i in range(len(coordinates1))}
    
    
    x2=np.zeros(len(ec2))
    y2=np.zeros(len(ec2))
    for i in range(len(ec2)):
        x2[i]=ec2[i][0]
        y2[i]=ec2[i][1]
    coordinates2 =ec2
    pos2 = {i: (coordinates2[i][0], coordinates2[i][1]) for i in range(len(coordinates2))}
    
    x3=np.zeros(len(ec3))
    y3=np.zeros(len(ec3))
    for i in range(len(ec3)):
        x3[i]=ec3[i][0]
        y3[i]=ec3[i][1]
    coordinates3 =ec3
    pos3 = {i: (coordinates3[i][0], coordinates3[i][1]) for i in range(len(coordinates3))}
    
    x4=np.zeros(len(ec4))
    y4=np.zeros(len(ec4))
    for i in range(len(ec4)):
        x4[i]=ec4[i][0]
        y4[i]=ec4[i][1]
    coordinates4 =ec4
    pos4 = {i: (coordinates4[i][0], coordinates4[i][1]) for i in range(len(coordinates4))}






    fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(20, 5))

    h = ax1.hist2d(x1, y1)
    fig.colorbar(h[3], ax=ax1)
    nx.draw(G1, pos=pos1,ax=ax1,alpha=0.4, node_size=10, node_color='red', edge_color='gray')
    ax1.set_title('WP')
    ax1.grid(True)

    h = ax2.hist2d(x2, y2)
    fig.colorbar(h[3], ax=ax2)
    nx.draw(G2,pos= pos2,ax=ax2,alpha=0.4, node_size=10, node_color='lightblue', edge_color='gray')
    ax2.set_title('PP')
    ax2.grid(True)
    
    h = ax3.hist2d(x3, y3)
    fig.colorbar(h[3], ax=ax3)
    nx.draw(G3,pos= pos3,ax=ax3,alpha=0.4, node_size=10, node_color='lightblue', edge_color='gray')
    ax3.set_title('PW')
    ax3.grid(True)
    
    h = ax4.hist2d(x4, y4)
    fig.colorbar(h[3], ax=ax4)
    nx.draw(G4,pos= pos4,ax=ax4,alpha=0.4, node_size=10, node_color='lightblue', edge_color='gray')
    ax4.set_title('WW')
    ax4.grid(True)

```

### Random initialization of epistemic space 

Initializing variables


```python
G1=initialize_network(number_of_authors,10)
G2=initialize_network(number_of_authors,10)
G3=initialize_network(number_of_authors,10)
G4=initialize_network(number_of_authors,10)
epistemic_space=initialize_epistemic_space(number_of_authors)
```

Heat map for the Epistemic space


```python
plot_epistemic_coverage(epistemic_space)
```


    
![png](https://github.com/devansh259/JKU_Journal_model/blob/main/Epistemic%20citation%20network%203/images/output_80_0.png)
    


Evolving networks using PP and WP models


```python
ec1=evolve_network_WP(G1,epistemic_space,time_steps,attempt_per_time_step)
ec2=evolve_network_PP(G2,epistemic_space,time_steps,attempt_per_time_step)

```


```python
ec3=evolve_network_PW(G3,epistemic_space,time_steps,attempt_per_time_step)
ec4=evolve_network_WW(G4,epistemic_space,time_steps,attempt_per_time_step)
```


```python
plot_epistemic_coverage(epistemic_space)
```


    
![png](output_84_0.png)
    


Degree Distribution


```python
degree_distribution_combined(G1,G2,G3,G4)
```


    
![png](output_86_0.png)
    


Epsitemic coverage heat map


```python
heatmap_combined(ec1,ec2,ec3,ec4)
```


    
![png](output_88_0.png)
    


Network in Epistemic space


```python
network_on_heatmap(G1,G2,G3,G4,ec1,ec2,ec3,ec4)

```


    
![png](output_90_0.png)
    


### Uniform initialization of epistemic space


```python
G1=initialize_network(number_of_authors,10)
G2=initialize_network(number_of_authors,10)
G3=initialize_network(number_of_authors,10)
G4=initialize_network(number_of_authors,10)
epistemic_space=initialize_agents_grid(number_of_authors)
```


```python
plot_epistemic_coverage(epistemic_space)
```


    
![png](output_93_0.png)
    



```python
ec1=evolve_network_WP(G1,epistemic_space,time_steps,attempt_per_time_step)
ec2=evolve_network_PP(G2,epistemic_space,time_steps,attempt_per_time_step)

```


```python
ec3=evolve_network_PW(G3,epistemic_space,time_steps,attempt_per_time_step)

```


```python
ec4=evolve_network_WW(G4,epistemic_space,time_steps,attempt_per_time_step)
```


```python
degree_distribution_combined(G1,G2,G3,G4)
```


    
![png](output_97_0.png)
    



```python
heatmap_combined(ec1,ec2,ec3,ec4)
```


    
![png](output_98_0.png)
    



```python
network_on_heatmap(G1,G2,G3,G4,ec1,ec2,ec3,ec4)
```


    
![png](output_99_0.png)
