"""
Monte Carlo multi area adequacy assessment case study
@author: Ensieh Sharifnia
Delft University of Technology
e.sharifnia@tudelft.nl

This code implements multi area adequacy assessment case study in the paper
"Generating Multivariate Load States Using a Conditional Variational Autoencoder",
Chenguang Wang, Ensieh Sharifnia, Zhi Gao, Simon H. Tindemans, Peter Palensky,
accepted for publication at PSCC 2022 and special issue of EPSR.
A preprint is available at: arXiv: 2110.11435
If you use (parts of) this code, please cite the preprint or published paper.
"""
# SPDX-License-Identifier: MIT

import numpy as np
import quadprog
import pandas as pd
from joblib import Parallel, delayed

import time
import os

# Define effective zero with room for plenty of rounding errors
EFFECTIVE_ZERO = 1e-5
class MAFApplication:
    '''
    Run Monte Carlo simulations to estimate resource adequacy risk indices.
    '''
    def __init__(self, unit_av = 0.83, unit_capacity=500, wind_precent = 0.05) -> None:
        '''
        Initialized network and matrices for quadratic solver.
        '''
        self.unit_availibility = unit_av
        self.unit_capacity = unit_capacity        
        self.wind_precent = wind_precent
        self.node, self.line = self.create_european_network()
        self.C_mat, self.flow_capacity = self.constraint_mat()

    def read_transfer(self, nodes, filepath):
        '''
        read transfer file for The 2025 forecasts of MAF2020
        Parameters:
            filepath: string
                    address of the transfer file.
        Returns:
            array of lines of the shape (2*L, 3) where L is the number of line
            each row represent an edge and columns show
            line[0]: nodeID from 
            line[1]: nodeID to 
            line[2]: capacity.
        '''
        tfdc = pd.read_excel(filepath, skiprows=9, sheet_name='HVDC')
        tfac = pd.read_excel(filepath, skiprows=9, sheet_name='HVAC' )
        maxfdc = tfdc.iloc[5:, 2:].max()
        maxfac = tfac.iloc[5:, 2:].max()
        maxf = maxfdc.add(maxfac, fill_value=0)

        from_c = [c[0:2] for c in maxf.index]
        to_c = [c[5:7] for c in maxf.index]
        maxf.index = pd.MultiIndex.from_arrays([from_c, to_c, maxf.index], names=('fromc','toc', 'area'))

        maxflow = maxf.groupby(level=["fromc", "toc"]).sum()
        maxflow.loc[ maxflow.index.get_level_values(0) == maxflow.index.get_level_values(1)]=np.nan
        maxflow.dropna(inplace=True)

        fromc = maxflow.index.get_level_values(0)
        toc = maxflow.index.get_level_values(1)
        lines = np.zeros((len(maxflow), 3))
        for index, name in enumerate(nodes['country']):
            fc = fromc == name
            tc = toc == name
            lines[fc, 0] = nodes['ID'][index]
            lines[tc, 1] = nodes['ID'][index]
        lines[:,2] = maxflow.to_numpy()
        return np.int32(lines)
    def smaller_network(self, select, nodes, lines):
        '''
        Build network based on the selected nodes
        Parameters:
            select: array
                Aarray of ID's for selected nodes.
            lines: array
                Array of edges
                Each row represent an edge and columns show
                line[0]: nodeID from 
                line[1]: nodeID to 
                line[2]: capacity. 
        Returns:
            New network based on the selected nodes
            snode: array
                Array of tuples with the size of select array (nodes of network).
                Each tuple contains:
                node[0]: country name
                node[1]: node ID
                node[2]: unit capacity MW
                node[3]: number of unit
            sline: array
                Array of lines of the shape (L, 3) where L is the number of lines (edges of network).
                Each row represent an edge and columns show
                line[0]: nodeID from 
                line[1]: nodeID to 
                line[2]: capacity.
        '''
        snode = nodes[select]
        f_ind = lines[:,1]==snode['ID'][0]
        t_ind = lines[:,0]==snode['ID'][0]
        for i in select:
            f_ind |= lines[:,1]==i
            t_ind |= lines[:,0]==i
        sline = lines[f_ind & t_ind]
        temp =  lines[f_ind & t_ind]
        for ind, v in enumerate(select):
            sline[temp[:,1]==v,1] = ind
            sline[temp[:,0]==v,0] = ind
            snode['ID'][ind] = ind

        return snode, sline
    def read_generation(self, filepath):
        '''
        read generation file for The 2025 forecasts of MAF2020
        Parameters:
            filepath: string
                    address of the generation file.
        Returns:
            node: array
            Array of tuples with the size of countries
            each tuple contains:
            node[0]: country name
            node[1]: node ID
            node[2]: unit capacity MW
            node[3]: number of unit
        '''        
        df = pd.read_csv(filepath,sep=', ',engine='python')
        list_index = {i:df.iat[i,0] for i in range(len(df))}
        df = df.rename(index=list_index)
        df = df.drop(columns=['Unnamed: 0'])
        area = df.columns
        gen_series = df.sum() - df.loc[['Wind Onshore', 'Wind Offshore', 'Solar (Photovoltaic)','Solar (Thermal)']].sum()
        wind_series = df.loc[['Wind Onshore', 'Wind Offshore']].sum()
        gen_series = gen_series.add(wind_series*self.wind_precent,fill_value = 0)
        country = [a[0:2] for a in area]
        gen_series.index = pd.MultiIndex.from_arrays([country, gen_series.index], names=('country', 'area'))
        gen_power = gen_series.groupby(level="country").sum()
        country = gen_power.index.values
        gen_power = gen_power.values
        
        num_unit = np.ceil(gen_power/self.unit_capacity)

        unit_capacity = self.unit_capacity + ((gen_power - num_unit*self.unit_capacity)/ num_unit)
        unit_capacity[num_unit == 0] = 0

        return np.rec.fromarrays((country,np.arange(len(country)), np.int32(unit_capacity), np.int32(num_unit)), names=('country','ID','unit_capacity','num_unit'))  
    def create_european_network(self):
        '''
        Build netwrok based on the considerations stated in section IV.B of the aformentioned paper.
        Returns:
            node : array of tuples with the size of 35
                each tuple represent a country and contains:
                node[0]: country name
                node[1]: node ID
                node[2]: unit capacity MW
                node[3]: number of unit
            line: array of lines of the shape (2*L, 3) where L is the number of line 
                each row represent an edge and columns show
                line[0]: nodeID from 
                line[1]: nodeID to 
                line[2]: capacity.
        '''
        file_path = '../data/MAF2020'
        dropcountry=[24,35]
        selected_countries = np.delete(np.arange(37), dropcountry, axis=0)
        nodes = self.read_generation(file_path+'/MAF_Generation_2025.csv')
        lines = self.read_transfer(nodes, file_path + '/Transfer_Capacities_MAF2020_Year2025.xlsx')
        # change unit capacity for Cypres
        nodes[6][2]=95
        nodes[6][3]=16
        return self.smaller_network(selected_countries, nodes, lines)
    def read_file(self, filepath):
        '''
        read file content
        Parameter:
            filepath: stirng
                file address.
        Return:
            data: array
                a numpy array of file content
        '''
        data = np.genfromtxt(filepath, delimiter=',')
        # remove inf, nan data if any
        data = data[np.isfinite(data).all(axis=1)]
        return data
    def set_demand(self, load_model):
        '''
        Set demand array for network based on the load_model.
        Parameter:
            load_model: string in {'historical_load', 'GANs', 'CVAE', 'VAE'}
                        name of load model
        Return:
            self.demand:  array
                        An array of demand [:,35]
        '''
        if (load_model == 'historical_load'):
            filepath = '../data/' + load_model + '.csv'
        else:
            filepath = '../data/process-data/generated-load/' + load_model + '.csv'
        if (load_model == 'GANs'):
            filepath = '../data/process-data/generated-load/' + load_model + '.npy'
            Gen_data_GANs = np.load(filepath)
            demand =Gen_data_GANs[:,0:35]
        elif (load_model == 'Gaussian_Copula'):
            demand = np.genfromtxt(filepath, delimiter=',')     
        elif (load_model == 'historical_load'):
            demand = np.genfromtxt(filepath, delimiter=',')[1:,:]
        else:
            demand = np.genfromtxt(filepath, delimiter=',')[1:,1:]
        demand = demand[(np.isfinite(demand) & np.array(demand>0)).all(axis=1)]
        self.demand = np.ma.round(demand)
        return self.demand
    def state_generator(self):
        '''
        Generate snapshot of network
        Returns: 
            state: array, size=[2, N] (N is the number of countries)
                A snapshot of network each column for one country and rows show
                state[0] : max available power
                state[1] : demand.
        '''
        state = np.zeros((2, self.node.shape[0]))
        state[1,:] = self.demand[np.random.choice(self.demand.shape[0]),:]
        for index, unit in enumerate(self.node):
            state[0, index] = unit[2]*np.sum(np.random.rand(unit[3]) <= self.unit_availibility*np.ones(unit[3]))
        state[0,state[0,:]<EFFECTIVE_ZERO] = 10 # prevent equality constrain
        return state
       
    def constraint_mat(self):
        '''build constraints matrix based on inequalities constraints equations: 9-11
        Returns:
            C_mat: array
                constraints matrix based on equations 9-11.
            flow_capacity: array, size [L,2] where L is the number of lines.
                capacity of each transmission line in each row, columns show
                flow_capacity[0]: forward flow
                flow_capacity[1]: backward flow
        '''
        n_lines = int(self.line.shape[0]/2)
        n_nodes = self.node.shape[0]
        grid = np.zeros(( n_lines, n_nodes))
        flow_capacity = np.zeros((n_lines, 2))
        index = 0
        for  line in (self.line):
            forward_line = ((grid[:,line[1]]==-1) & (grid[:,line[0]]==1))
            if np.sum(forward_line)>0:
                flow_capacity[forward_line,1] = line[2]
            else:
                grid[index, line[0]] = -1
                grid[index, line[1]] = 1
                flow_capacity[index, 0] = line[2]
                index += 1

        cineq = np.vstack((np.zeros((n_lines, 2*n_nodes)), 
                           np.hstack((np.identity(n_nodes), -np.identity(n_nodes) )) ))
        
        fineq = np.vstack((np.hstack((np.identity(n_lines), -np.identity(n_lines) )),
                        np.zeros((n_nodes, 2*n_lines)) ))  
        nineq = np.hstack((np.vstack((grid, np.identity(n_nodes))), -np.vstack((grid, np.identity(n_nodes))) ))
        
        return np.hstack(( cineq, fineq, nineq)), flow_capacity
    
    def solving_quad(self, state):
        '''
        use inequalities (9-11)
        and solving this quadratic optimization (8)
        objective function: Minimize     1/2 x^T G x - a^T x
                            Subject to   C.T x >= b
                            where x is vector = [lines flow, load curtailment in each node]
        Parameter:
            state: array, size=[2, N] (N is the number of countries)
                A snapshot of network each column for one country and rows show
                state[0] : max available power
                state[1] : demand.
        Returns:
            node_loss: an array of size [N,]
                   load curtailments of the countries according to the input state.        
        '''
        n_lines = int(self.line.shape[0]/2)
        n_nodes = self.node.shape[0]
        eps = 1e-10  # NOTE: this should be zero, but the solver requires positive definite G
        # only loss of load contribute in minimization of objective function

        G_mat = np.vstack((np.hstack(( np.diag(np.ones(n_lines)*eps) , np.zeros((n_lines, n_nodes)))),
                            np.hstack((np.zeros((n_nodes , n_lines )) , np.diag(1/state[1,:]))) ))
        a_vec = -np.hstack((np.ones(n_lines)*eps ,  np.ones(n_nodes)))

        b_vec = np.zeros(self.C_mat.shape[1])
        b_vec[n_nodes:2*n_nodes] = - state[1,:] # c constrains

        b_vec[2*n_nodes:2*n_nodes+n_lines] = -self.flow_capacity[:, 1] # -f_k^max =< f_k
        b_vec[2*n_nodes+n_lines : 2*n_nodes+2*n_lines] = -self.flow_capacity[:,0] # -f_k^max =< -f_k
        b_vec[2*n_lines+2*n_nodes : 2*n_lines+3*n_nodes] = state[1,:] -state[ 0,:] # flow<= demand-g^max
        b_vec[-n_nodes:] = -state[1,:] # flow<-demand

        try:
            node_loss = quadprog.solve_qp(G=G_mat, a=a_vec, C=self.C_mat, b=b_vec,
                                           factorized=False)[0][-n_nodes:]
        except Exception as e: 
            print(e)
            print(state)
            x_var = self.C_mat[:,0] # zeros for transfer capacity
            x_var[-n_nodes:] = np.maximum(np.zeros(n_nodes), state[1,:] -state[ 0,:] )  # demand-generation
            constraints = (np.dot(self.C_mat.T, x_var)<b_vec)
            print(f"#constraints are inconsistent {np.sum(constraints)}, #all constraints {len(b_vec)}" )
            return x_var[-n_nodes:]
        node_loss[node_loss < 0.0001] = 0
        return node_loss
    def innerloop_MCS(self):
        '''
        Generate a state and compute Loss Of Load and Energy Not Served Values for that
        Returns:
            LOL_no_connection: Array of the shape (N) (N is the number of countries)  
                Loss Of Load for the generated state if there is not any connection between countries.
            ENS_no_connection: Array of value of shape (N) (N is the number of countries)
                Energy Not Served for the generated state  if there is not any connection between countries.
            LOL: Array of the shape (N) (N is the number of countries)  
                Loss Of Load for the generated state.
            ENS: Array of value of shape (N) (N is the number of countries)
                Energy Not Served for the generated state.
        '''
        state = self.state_generator()            
        load_curtialment = self.solving_quad(state)
        load_curtialment_no_connection = np.maximum(0, state[1]-state[0])
        return 8760 * (load_curtialment_no_connection>EFFECTIVE_ZERO), 8760 * load_curtialment_no_connection, 8760 * (load_curtialment>EFFECTIVE_ZERO), 8760 * load_curtialment
        
    def MCS(self, num_itr, load_model, joblib=True):
        '''
        Run Monte Carlo simulation
        Parameters:
            num_itr: int
                Number of iteration for simulation
            load_model: {'historical_load', 'GANs', 'CVAE', 'VAE','Gaussian-Copula'}
                Name of load model generator.  
            joblib: bool, optional
                If True, then Use all cores (parallel processing)., otherwise use one core. The default value of joblib is True.
        Returns:
            lol_no_connection: array of float
                Array of Loss Of Load of shape (num_itr, N) (N is the number of countries).  if there is not any connection between countries.
            ens_no_connection: array of float
                Array of Energy Not Served of shape (num_itr, N) (N is the number of countries).  if there is not any connection between countries.
            lol: array of float
                Array of Loss Of Load of shape (num_itr, N) (N is the number of countries) 
            ens: array of float
                Array of Energy Not Served of shape (num_itr, N) (N is the number of countries)
            execution_time: float
                duration of simulation in ms
        '''
        
        self.set_demand(load_model=load_model)        
        execution_time = time.time()
        if joblib:
            backend = 'loky'
            job_results = Parallel(n_jobs=-1, backend=backend)(delayed(
                self.innerloop_MCS)() for _ in range(num_itr))
            l_values_no, e_values_no, l_values, e_values = zip(*job_results)
            lol_no_connection = np.asfarray(l_values_no)
            ens_no_connection = np.asfarray(e_values_no)
            lol = np.asfarray(l_values)
            ens = np.asfarray(e_values)
        else:
            lol = np.zeros((num_itr, self.node.shape[0]))
            ens = np.zeros((num_itr, self.node.shape[0]))
            lol_no_connection = np.zeros((num_itr, self.node.shape[0]))
            ens_no_connection = np.zeros((num_itr, self.node.shape[0]))            
            for i in range(num_itr):
                lol_no_connection[i,:], ens_no_connection[i, :], lol[i,:], ens[i, :] = self.innerloop_MCS()
        execution_time = time.time() - execution_time
        return lol_no_connection, ens_no_connection, lol, ens, execution_time
    def expectation_err(self, risk, risk_measures, percountry):
        '''
        This function compute the risk expectation and standard error of estimation
        Parameters:
            risk: string ['LOLE', 'EENS']
                compute LOLE or EENS risk index
            risk_measures: Array 
                        size(num_iterations, N) N is the number of countries
                        values of risk per state and countries
            percountry: bool
                        f True, then compute risk index per country. otherwise consider all countries together.
        Returns:
            risk_expectation: array or scaler
                            risk estimation. Array for per country, scaler for other
            std_error: array or scaler
                    standard error of estimation. Array for per country, scaler for other
         
        '''
        if percountry:
            risk_expectation = np.mean( risk_measures, axis = 0)
            std_error = np.std(risk_measures, axis=0, ddof = 1)/ np.sqrt(risk_measures.shape[0])
        else:
            if (risk == 'LOLE'):
                q_measure = np.max(risk_measures, axis = 1)
            else:
                q_measure = np.sum(risk_measures, axis = 1)
            risk_expectation = np.mean(q_measure)
            std_error = np.std(q_measure, ddof=1)/ np.sqrt(q_measure.shape[0])
        return risk_expectation, std_error
    def scientific_notation(self, risk_value, std_error):
        '''
        providing scientific notation for risk value along with standard deviation
        Parameters:
            risk_value: float
            std_error: float
        Returns:
            string of scientific notation
        '''
        if std_error < EFFECTIVE_ZERO and risk_value< EFFECTIVE_ZERO:
            return '{}({})'.format(int(risk_value), int(std_error))
        std_error_e = "{:.5e}".format(std_error)
        risk_e = "{:.10e}".format(risk_value)
        std_error_power = int(std_error_e[std_error_e.find('e')+1:])
        risk_power = int(risk_e[risk_e.find('e')+1:])
        precision = risk_power - std_error_power
        
        std_e = int(np.round(float(std_error_e[0:std_error_e.find('e')])))
        if (std_e < 3):
            precision +=1
            std_error_e = "{:.2e}".format(std_error)
            std_e = int(np.round(float(std_error_e[0:std_error_e.find('e')])*10))
        f = "{:."+str(precision)+"e}"
        risk_e = f.format(risk_value)
        
        if risk_power>3 or risk_power<-3 :
            return '{}({})x10^{}'.format(risk_e[0:risk_e.find('e')], std_e, risk_power)
        elif risk_power<4 and std_error_power>=0:
            if std_error_power==0 and int(np.round(std_error))<3 :
                f = "{:.1e}"
                risk_e = float(risk_e)
            else:
                std_e = int(np.round(std_error))
                risk_e = int(np.round(risk_value))
            return "{}({})".format(risk_e, std_e)
        else:
            return "{}({})".format(float(risk_e), std_e)

    def print_results_countries(self,risk, risk_measures, percountry, country_list = None):
        '''
        this function print the risk expectation +- stdandard error
        Parameters:
            risk: {'LOLE', 'EENS'}
                name of risk measure
            risk_measures: an array [num_iteration, num_countries]
                an array of risk measurements for each instance in Monte Carlo simulation.
            percountry: Bool (optional)
                if True, then mean value and standard error are printed for each country.
                Otherwise, consider all countries as an area and print the risk value and corresponding standard error.      
            country_list: array (optional)
                An array of countries ID. percountry should be Ture                 
                if provided, then print mean value and standard error for countries in country_list.
                otherwise, mean value and standard error are printed for all countries.
        '''
        risk_expectation, std_error = self.expectation_err(risk, risk_measures, percountry)
        print(risk)
        if percountry: 
            if country_list is None:       
                for i, c in enumerate(self.node['country']):
                    print(f"{c}: {self.scientific_notation(risk_expectation[i],std_error[i])}")
            else:
                for i in country_list:
                    print(f"{self.node['country'][i]}: {self.scientific_notation(risk_expectation[i],std_error[i])}")
        else:
            print(f"{risk_expectation}+-{std_error}")
if __name__ == "__main__":
    maf_obj = MAFApplication()
    number_iterations = 100
    load_models =[ 'historical_load', 'CVAE', 'VAE','Gaussian_Copula', 'GANs']
    #AT, NL, UK
    country_list = [1, 24, 34]

    for i , load_model in enumerate( load_models):

        (MC_lol_no_connection, MC_ens_no_connection, MC_lol,MC_ens, MC_time) = maf_obj.MCS( num_itr=number_iterations, load_model=load_model, joblib=True) 
        print(f"================={load_model}============================")
        maf_obj.print_results_countries(risk='LOLE', risk_measures=MC_lol, percountry=True, country_list=country_list)
        maf_obj.print_results_countries(risk='EENS', risk_measures=MC_ens, percountry=True, country_list=country_list)       
        print(f"================={load_model} islanded ===================")
        maf_obj.print_results_countries(risk='LOLE', risk_measures=MC_lol_no_connection, percountry=True, country_list=country_list)
        maf_obj.print_results_countries(risk='EENS', risk_measures=MC_ens_no_connection, percountry=True, country_list=country_list)

        