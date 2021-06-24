#!/bin/env python
# code on github : https://github.com/lujunyan1118/SimpleBool

'''
   SimpleBool is a python package for dynamic simulations of boolean network 
    Copyright (C) 2013  Junyan Lu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

#from __future__ import division
#__metaclass__=type

import random
import sys
import os
import shutil
import datetime

#=======================================================================================================================
class BooleanModel:
#=======================================================================================================================

    '''
    random.seed()    
    KEEP_CONSTANT={}
    INITIAL_VALUES={}
    REGULATOR_NODES=[]
    TRUTH_TABLE=[]
    MAPPING_TABLE={}
    SIM_PARAMS={}
    '''

    #=======================================================================================================================
    def __init__(self, arg_path_to_inifile, mut_nodes=[]):
    #=======================================================================================================================
        random.seed()    
        self.KEEP_CONSTANT = {}
        self.INITIAL_VALUES = {}
        self.REGULATOR_NODES = []
        self.TRUTH_TABLE = []
        self.MAPPING_TABLE = {}
        self.SIM_PARAMS = {}
        self.FINAL_VALUES = {}

        self.arg_path_to_inifile = arg_path_to_inifile
        self.root_working_folder = os.getcwd()
        self.outputs_folder_path = ""

        SIM_PARAMS = self.parse_all_params(arg_path_to_inifile)

        self.SIM_PARAMS = SIM_PARAMS

        for on_nodes in SIM_PARAMS['initially_on']:
            self.INITIAL_VALUES[on_nodes] = True
            
        for off_nodes in SIM_PARAMS['initially_off']:
            self.INITIAL_VALUES[off_nodes] = False

        for on_nodes in SIM_PARAMS['always_on']:
            self.INITIAL_VALUES[on_nodes] = True
            self.KEEP_CONSTANT[on_nodes] = True

        for off_nodes in SIM_PARAMS['always_off']:
            self.KEEP_CONSTANT[off_nodes] = False
            self.INITIAL_VALUES[off_nodes] = False
                    
        for node, state in mut_nodes:
            self.KEEP_CONSTANT[node] = state
            self.INITIAL_VALUES[node] = state

        try:
            (self.REGULATOR_NODES, self.TRUTH_TABLE, self.MAPPING_TABLE) = self.get_all_expressions_truth_table(open(SIM_PARAMS['rules_file_path']).read(), self.KEEP_CONSTANT)
        except Exception as e:
            print("\n\n---> got an exception in get_all_expressions_truth_table()...")
            sys.exit(18)

        model_verbose = {'SYNC':'Synchronous','GASYNC':'General Asynchrounous','ROASYNC':'Random Order Asynchrounous'}
        print('''BooleanModel initialization completed!                    
            Total nodes number:    %s
            Simulation num_steps:    %s
            Simulation num_rounds:    %s
            Simulation mode:    %s
            '''%(len(self.MAPPING_TABLE.keys()),SIM_PARAMS['num_steps'],SIM_PARAMS['num_rounds'],model_verbose[SIM_PARAMS['sim_mode']]))


    # =======================================================================================================================
    def parse_all_params(self, arg_path_to_inifile):
    # =======================================================================================================================

        '''#parser parameters for simulation and transition matrix building'''
        SIM_PARAMS = {'rules_file_path': 'rules.txt',
                      'initially_on': '',
                      'initially_off': '',
                      'always_on': '',
                      'always_off': '',
                      'num_rounds': 1,
                      'num_steps': 1,
                      'sim_mode': 'SYNC',  # we allow 3 modes : {SYNC, GASYNC, ROASYNC}
                      'plot_nodes': '',
                      'init_if_missing': 'random',
                      'output_folder_base_path': ''
                      }  # define parameters

        # relative to absolute path conversion
        path_to_inifile = str(arg_path_to_inifile)
        if (path_to_inifile.startswith('./')):
            path_to_inifile = path_to_inifile[len('./'):]
        path_to_inifile = os.path.join(self.root_working_folder, path_to_inifile)

        for each_line in open(path_to_inifile).readlines():
            if each_line.strip().startswith("#"): continue
            param_name = each_line.split('=')[0].strip()
            param_value = each_line.split('=')[1].strip()
            param_value = param_value.split("#")[0].strip()

            if param_name in SIM_PARAMS.keys():
                SIM_PARAMS[param_name] = param_value
            else:
                print("Error: skipping unknown parameter: %s" % param_name)
                pass

        # formalize parameters
        try:
            SIM_PARAMS['rules_file_path'] = str(SIM_PARAMS['rules_file_path'])
            SIM_PARAMS['initially_on'] = [node.strip() for node in SIM_PARAMS['initially_on'].split(',')]
            SIM_PARAMS['initially_off'] = [node.strip() for node in SIM_PARAMS['initially_off'].split(',')]
            SIM_PARAMS['always_on'] = [node.strip() for node in SIM_PARAMS['always_on'].split(',')]
            SIM_PARAMS['always_off'] = [node.strip() for node in SIM_PARAMS['always_off'].split(',')]
            SIM_PARAMS['plot_nodes'] = [node.strip() for node in SIM_PARAMS['plot_nodes'].split(',')]
            SIM_PARAMS['num_rounds'] = int(SIM_PARAMS['num_rounds'])
            SIM_PARAMS['num_steps'] = int(SIM_PARAMS['num_steps'])
            SIM_PARAMS['sim_mode'] = str(SIM_PARAMS['sim_mode'])
            SIM_PARAMS['init_if_missing'] = {'random': 'random', 'True': 1, 'true': 1, 'False': 0, 'false': 0}[str(SIM_PARAMS['init_if_missing'])]

            # relative to absolute path conversion
            output_folder_base_path = str(SIM_PARAMS['output_folder_base_path'])
            if(output_folder_base_path.startswith('[work_root_folder]/')) :
                output_folder_base_path = output_folder_base_path[len('[work_root_folder]/'):]
            output_folder_base_path = os.path.join(self.root_working_folder, output_folder_base_path)
            #if output_folder_base_path and (not output_folder_base_path):
            #    os.mkdir(output_folder_base_path)
            SIM_PARAMS['output_folder_base_path'] = output_folder_base_path

            # relative to absolute path conversion
            rules_file_path = SIM_PARAMS['rules_file_path']
            if (rules_file_path.startswith('[work_root_folder]/')):
                rules_file_path = rules_file_path[len('[work_root_folder]/'):]
            rules_file_path = os.path.join(self.root_working_folder, rules_file_path)
            SIM_PARAMS['rules_file_path'] = rules_file_path

            # check empty params.
            for key in SIM_PARAMS.keys():
                if SIM_PARAMS[key] == ['']:
                    SIM_PARAMS[key] = []

        except:
            print("Error: Invalid input data types!")

        if SIM_PARAMS['sim_mode'] not in ['GASYNC', 'SYNC', 'ROASYNC']:
            print("\n\n---> sim_mode = %s" % SIM_PARAMS['sim_mode'])
            print("---> wrong simulation mode parameter provided! supported modes are: 'GASYNC', 'SYNC','ROASYNC'")
            print("---> please fix sim_mode in your simulation .ini file!")
            sys.exit(18)

        try:  # create a timestamped output folder for this simulation.
            now = datetime.datetime.now()
            date_time_str_format = "%Y_%m_%d__%H%M%S"
            output_folder_name = "sim_outputs_" + now.strftime(date_time_str_format)
            outputs_path = os.path.join(SIM_PARAMS['output_folder_base_path'], output_folder_name)

            if outputs_path and (not os.path.exists(outputs_path)):
                os.makedirs(outputs_path)

            self.outputs_folder_path = outputs_path

            # copy the rules.txt + .ini file to the output folder, so we have "all package" to restore run if we want.
            shutil.copy(SIM_PARAMS['rules_file_path'], outputs_path)
            shutil.copy(path_to_inifile, outputs_path)

        except Exception as e:
            print("\n\n---> Failed to create outputs folder at path = %s !!" % outputs_path)

        return SIM_PARAMS


    # =======================================================================================================================
    def get_expression_nodes(self, expression):
    # =======================================================================================================================
        '''convert one line of expression to a node list'''
        nodes = []
        other = ['=', 'and', 'or', 'not']  # remove operator signs
        for node in expression.split():
            node = node.strip('*() ')  # remove * ( ) from the node name
            if node not in other:
                nodes.append(node)
        return nodes


    # =======================================================================================================================
    def get_expression_truth_table(self, expression, dict_nodes_to_keep_constant):
    # =======================================================================================================================

        '''Iterate all the state of input node and output all the inital state and the value of the target node,
        used to construct truth table.
        Return a list of tuples, the first tuple contain the name of target node and its regulators,
        the rest of the tuple contain all the possible state of the target node and its regulators,
        the first element in the tuple is the state of target'''

        nodes = self.get_expression_nodes(expression)

        record = []  # to store results
        all_regulators = nodes[1:]  # all regulators of the target (the components affecting the target node)
        target_node = nodes[0]
        bool_func = expression.split('=')[1].strip()

        # record the target node and its regulators in a tuple (Target, R1, R2, ...,Rn)
        record.append(tuple([target_node] + all_regulators))

        # for each regulator of this node that needs to be set with constant value - set it here.
        for node in set(all_regulators) & set(dict_nodes_to_keep_constant.keys()):
            vars()[node] = dict_nodes_to_keep_constant[node]  # set the value of the constant nodes

        num_inp_combinations = 2 ** len(all_regulators)  # n nodes have 2**n combinations

        # state is the T/F value of all regulators. So for N regulators we have a 2^N bit-vector.
        # go over all state combinations for the N regulators.
        for index in range(num_inp_combinations):

            # bin(index) returns string that starts with "0b" so skip the first 2 chars.
            # state will be a bitvector string of length len(all_regulators) e.g. 00000100100001..0
            state = bin(index)[2:].zfill(len(all_regulators))

            # set all regulators with their value as in current state
            for r in range(len(all_regulators)):
                vars()[all_regulators[r]] = int(state[r])

            # calculate the target node value w.r.t. its boolean func. Or to its constant value if in constant nodes list.
            if target_node not in dict_nodes_to_keep_constant:
                target_val = int(eval(bool_func))
            else:
                target_val = int(dict_nodes_to_keep_constant[target_node])

            # each entry in the record table will consist a tuple (y, in1,in2,in3,....,inN)
            # where y is the output of the boolean func given the state input (in1,in2,...inN).
            record.append(tuple([target_val] + [int(n) for n in state]))

        return record

    # =======================================================================================================================
    def get_all_expressions_truth_table(self, booltext, dict_nodes_to_keep_constant):
    # =======================================================================================================================

        '''Construct the truth table that contain all the possible input state and output state for each node'''
        all_results = []
        all_nodes = set([])  # all nodes in boolean rule file
        target_nodes = set([])  # all target nodes in boolean rule file (i.e. they have regulators)
        regulator_nodes_table = []  # a list contain regulator of each node as a tuple. The tuple index in the list is the target node index
        truth_table = []  # a list of dictionary contain the truth table for each node. The sequence is in consist with node sequence in mapping

        for line in booltext.split('\n'):

            if line.strip() != '' and line[0] != '#':
                line_nodes = self.get_expression_nodes(line)
                target_nodes = target_nodes | set([line_nodes[0]])
                all_nodes = all_nodes | set(line_nodes)
                if line_nodes[0] not in dict_nodes_to_keep_constant.keys():
                    try:
                        all_results.append(self.get_expression_truth_table(line, dict_nodes_to_keep_constant))
                    except:
                        print("\n\n---> Error : failed in get_expression_truth_table() of line = %s" % line)
                else:
                    # if node X is in dict_nodes_to_keep_constant, then its bool_func is X=X, so it's regulator is X and its
                    # output is X, and the combinations are
                    # (out, inp) = (1,1) and (0,0) only.
                    all_results.append([(line_nodes[0], line_nodes[0]), (1, 1), (0, 0)])

        # for each node that is not mapped (i.e. not a target node) - use the regulation rule X=X.
        unmapped_nodes = all_nodes - target_nodes
        for unmapped_node in unmapped_nodes:
            all_results.append([(unmapped_node, unmapped_node), (1, 1), (0, 0)])

        # sort the table by regulator names
        sorted_all = sorted(all_results, key=lambda x: x[0][0])

        # map node names N1, N2, N3,... to integers 0,1,2,3,...
        mappings_table = dict(zip([node[0][0] for node in sorted_all], range(len(sorted_all))))

        # generate list of regulators for each node and the truth table.
        for each_node in sorted_all:

            state_dict = {}
            # regulators will be tuple of indexes (as in mappings_table)
            regulators = tuple([mappings_table[node] for node in each_node[0][1:]])
            regulator_nodes_table.append(regulators)

            # for each state combination of target node - set its output.
            # so per each target node we create a dictionary of the form:
            # {(0,0,0,0,..) : 1,
            # (1,0,0,0,..) : 0,
            # (0,1,0,0,..) : 1,
            # etc...}
            for each_state in each_node[1:]:
                state_dict[each_state[1:]] = each_state[0]

            truth_table.append(state_dict)

        return regulator_nodes_table, truth_table, mappings_table

    #=======================================================================================================================
    def get_sim_mode(self):
    #=======================================================================================================================

        return self.SIM_PARAMS['sim_mode']

    #=======================================================================================================================
    def get_outputs_path(self):
    #=======================================================================================================================

        return self.outputs_folder_path

    #=======================================================================================================================
    def get_sorted_node_names(self):
    #=======================================================================================================================

        return sorted(self.MAPPING_TABLE)

    #=======================================================================================================================
    def save_final_nodes_value(self, file_out = 'final_node_values.txt'):
    #=======================================================================================================================

        all_nodes = self.get_sorted_node_names()
        on_nodes=[]
        off_nodes=[]
        output=file(file_out,'w')
        for node in all_nodes:
            output.writelines('%s\t%s\n'%(node,self.FINAL_VALUES[node]))
            if self.FINAL_VALUES[node] == 0:
                off_nodes.append(node)
            elif self.FINAL_VALUES[node] == 1:
                on_nodes.append(node)
        print('''%s nodes stabilized on 'ON' state: %s '''%(len(on_nodes),','.join(on_nodes)))
        print('''%s nodes stabilized on 'OFF' state: %s '''%(len(off_nodes),','.join(off_nodes)))

        output.close()

    #=======================================================================================================================
    def run_one_iteration_sync(self, initial_state):
    #=======================================================================================================================
        '''Iterate model using sychronous method. The most time consuming part, need to be carefully optimized'''
        # go over all nodes n, for each node set its regulator values (w.r.t. initial_state) - and calculate its output
        # by using the TRUTH_TABLE.
        # go over all nodes, and for each_node (n) do:
        # go over each of its each_regulator (r) and create a tuple (r1,r2,...rN) with the value of each regulator rj
        # to be as in the given initial_state.
        # once you have the new system state - simply go to TRUTH_TABLE[each_node][new state].
        # so in simple words its doing : for each node, set the value of its regulators w.r.t. current_state, and then
        # go to the TRUTH_TABLE to calculate output per each node, and combine the new system state.
        #
        new_state = [str(self.TRUTH_TABLE[n][tuple([int(initial_state[r]) for r in self.REGULATOR_NODES[n]])]) for n in range(len(initial_state))]
        return ''.join(new_state)

    #=======================================================================================================================
    def run_one_iteration_gasync(self, initial_state):
    #=======================================================================================================================

        '''Iterate model using asynchronous method (General Asynchronous model: update one random node per step)'''
        # with ASYNC mode - only 1 node will be executed and updated (all rest nodes, including its dependent will not change!).
        # so what we do is:
        # (1) first we randomly select index n of node to be updated.
        # (2) then we set all its regulator values w.r.t. initial_state.
        # (3) then we use the TRUTH_TABLE of node n - to check for its new value given its regulator values.
        #     and we update new_state ONLY in the index of that node n.
        #
        updated_node_n = random.randint(0, len(initial_state)-1)
        new_state = list(initial_state)
        new_state[updated_node_n] = str(self.TRUTH_TABLE[updated_node_n][tuple([int(initial_state[r]) for r in self.REGULATOR_NODES[updated_node_n]])])
        return ''.join(new_state)

    #=======================================================================================================================
    def run_one_iteration_roasync(self, initial_state):
    #=======================================================================================================================

        # with RANDOM ORDER ASYNC mode - we update all nodes sequantially but in random order.
        # so what we do is:
        # (1) first we randomly select a sequence order for the update.
        # (2) then we go over all nodes n by order in seq - and set all its regulator values w.r.t. current state.
        # (3) then we use the TRUTH_TABLE of node n - to check for its new value given its regulator values.
        #     and we update new_state for each node n.
        # NOTE - that this is different from SYNC mode where all nodes updated at once! here we update sequentially,
        # and each update affects for the next update.
        #
        seq = range(len(initial_state))
        random.shuffle(seq)  # generate a random sequence of updating list
        new_state = list(initial_state)
        for n in seq:
            new_state[n]= str(self.TRUTH_TABLE[n][tuple([int(new_state[r]) for r in self.REGULATOR_NODES[n]])])
            #new_state = [str(self.TRUTH_TABLE[n][tuple([int(new_state[r]) for r in self.REGULATOR_NODES[n]])]) for n in seq]
        return ''.join(new_state)

    #=======================================================================================================================
    def get_initial_state(self, init_if_missing='random'):
    #=======================================================================================================================

        initial_state = []

        for node in sorted(self.MAPPING_TABLE.keys()):
            if node in self.INITIAL_VALUES:
                initial_state.append(str(int(self.INITIAL_VALUES[node])))
            else:
                if init_if_missing == 'random':
                    initial_state.append(random.choice(['0','1']))
                else:
                    # here init_if_missing is True or False.
                    initial_state.append(str(int(init_if_missing)))

        return ''.join(initial_state) # so state is a tring e.g. '0100111100...'

    #=======================================================================================================================
    def add_string(self, xlist, ystring):
    #=======================================================================================================================
        return [x + int(y) for x, y in zip(xlist, ystring)]

    #=======================================================================================================================
    def calc_average(self, x):
    #=======================================================================================================================

        return x/self.num_rounds

    #=======================================================================================================================
    def run_model_simulation(self):
    #=======================================================================================================================
        
        traj_all = []
        num_steps = self.SIM_PARAMS['num_steps']
        num_rounds = self.SIM_PARAMS['num_rounds']
        init_if_missing = self.SIM_PARAMS['init_if_missing']

        self.num_rounds = num_rounds

        # create a list of lists : for each step - we have state of all target nodes.
        # for example for 4 target nodes and 3 steps we'll have :
        # collect = [
        #             [0,1,0,1]  --> for step 0
        #             [1,1,1,0]  --> for step 1
        #             [0,0,0,0]  --> for step 2
        #             ]
        collect = [[0]*len(self.MAPPING_TABLE)]*(num_steps+1)

        # round loop - we run NROUNDS and AVERAGE at the end over the rounds.
        for r in range(num_rounds):

            traj = []
            initial_state = self.get_initial_state(init_if_missing)
            traj.append(initial_state)
            current = initial_state

            collect[0] = self.add_string(collect[0], current)

            # steps loop - we run the rules NSTEP times (so y=f(inp) NSTEP times).
            for s in range(num_steps):

                if self.SIM_PARAMS['sim_mode'] == 'SYNC':
                    next = self.run_one_iteration_sync(current)
                elif self.SIM_PARAMS['sim_mode'] == 'GASYNC':
                    next = self.run_one_iteration_gasync(current)
                elif self.SIM_PARAMS['sim_mode'] == 'ROASYNC':
                    next = self.run_one_iteration_roasync(current)

                traj.append(next)

                # we want to AVERAGE the results over NROUNDS so sum here in add_string.
                collect[s+1] = self.add_string(collect[s+1], next)

                current = next

            traj_all.append(traj)

        final_outputs = {}

        # average all the rounds per step.
        averaged_outputs = [list(map(self.calc_average, each_step)) for each_step in collect]

        # so averaged_outputs is a "table" of size N_STEPS x N_NODES:
        # averaged_outputs = [
        #             [0,1,0,1, ... ,nodeN_value]  --> for step 0
        #             [1,1,1,0, ... ,nodeN_value]  --> for step 1
        #             [0,0,0,0, ... ,nodeN_value]  --> for step 2
        #             ]
        #

        # print("\n\n-----> averaged_outputs size : NSTEPS = %s, NNODES = %s\n\n" % (len(averaged_outputs), len(averaged_outputs[0])))

        nodes_list = self.get_sorted_node_names()

        # per each node - collect all its steps state and put them in dict().
        # so we'll have something like this :
        # final_outputs = { AKT : [0,1,0,....., stepN],
        #                   APC : [1,0,1,....., stepN],
        #                   ASK1: [1,1,0,....., stepN],
        #                   ...}
        # self.FINAL_VALUES will be dict() with just the final state per each node :
        # self.FINAL_VALUES = { AKT : 1,
        #                APC : 0,
        #                ASK1: 0,
        #                ...}

        for node_i in range(len(nodes_list)):
            final_outputs[nodes_list[node_i]] = [steps_state[node_i] for steps_state in averaged_outputs]
            self.FINAL_VALUES[nodes_list[node_i]] = final_outputs[nodes_list[node_i]][-1]

        return final_outputs

#=======================================================================================================================
#=======================================================================================================================
# Utility methods below
#=======================================================================================================================
#=======================================================================================================================

#=======================================================================================================================
def plot_result(results,plotlist,marker=True):
#=======================================================================================================================

    import matplotlib.pyplot as plt
    
    '''Plot the simulated results'''
    print("Ploting results...")

    plotsymbyl=['o','v','*','s','+','p','x','1','2','h','D','.',','] # plot line with symbyl
    ploti=0
    for items in plotlist:  # plot nodes states using matplotlib
        if marker:
            plt.plot(results[items],label=items,linewidth=2.5,linestyle='-',marker=plotsymbyl[ploti]) #with marker
        else: plt.plot(results[items],label=items,linewidth=2.5,linestyle='-') #no marker
        
        ploti += 1
        if ploti >= 12: ploti=0

    plt.xlabel('Steps',size=15)
    plt.ylabel('Percentage',size=15)
    plt.yticks([-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1],size=15)
    plt.xticks(size=15)
    plt.legend(prop={'size':15}) # make legend
    plt.show()  # show plot

    # plt.savefig('figure.png',dpi=300)

    return

#=======================================================================================================================
def write_data(results, outputs_path='', saved_filename='output_data.txt', sample_step=1):
#=======================================================================================================================

    save_fullpath = os.path.join(outputs_path, saved_filename)

    outfile_obj = open(save_fullpath,'w')

    for nodes in sorted(results):
        outfile_obj.writelines('%-15s' % nodes)
        for data_values in results[nodes][1::sample_step]:
            outfile_obj.writelines('%-8.2f' % data_values)
        outfile_obj.writelines('\n')

    outfile_obj.close()

#=======================================================================================================================
def my_grid_plot(final_results_all_steps, sim_mode_str='', thresh=0.5, 
                 outputs_path='', saved_filename='sim_summary_fig.svg'):
#=======================================================================================================================

    import matplotlib.pyplot as plt
    from matplotlib import colors
    import numpy as np

    figure_save_fullpath = os.path.join(outputs_path, saved_filename)

    list_keys = list(final_results_all_steps.keys())
    num_nodes = len(list_keys)
    # list_nodes_ints = list(range(0, num_nodes))
    # list_nodes_ints_str = map(str, list_nodes_ints)

    num_steps = len(final_results_all_steps[list_keys[0]])
    list_steps = list(range(0, num_steps))
    list_steps_str = map(str, list_steps)

    sanity_test = False

    if not sanity_test:

        data_array = np.random.rand(num_nodes, num_steps)

        for k in range(num_nodes):
            node = list_keys[k]
            data_array[k, :] = final_results_all_steps[node]
    else:
        num_steps = 7
        num_nodes = 4
        list_keys = ['A','B23','C234','D5PTK']
        list_steps = list(range(0, num_steps))
        list_steps_str = map(str, list_steps)

        data_array = np.random.rand(num_nodes, num_steps)
        data_array[0, :] = [0, 0.2, 0.34, 0.1, 0.04, 0.335, 0.8]
        data_array[1, :] = [0, 0.2, 0.34, 0.1, 0.04, 0.335, 0.8]
        data_array[2, :] = [0, 0.2, 0.34, 0.1, 0.04, 0.335, 0.8]
        data_array[3, :] = [0, 0.2, 0.34, 0.1, 0.04, 0.5, 1.0]


    if True:
        # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
        #data_array = np.random.rand(num_nodes, num_steps)
        #cmap = plt.cm.RdBu
        #cmap = 'Reds'
        #cmap = plt.cm.jet
        plt.imshow(data_array, cmap=plt.cm.jet, interpolation='nearest')
        fig = plt.gcf()
        ax = plt.gca()

        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.2)

        ax.set_xticks(np.arange(-.5, num_steps, 1))
        ax.set_yticks(np.arange(-.5, num_nodes, 1))

        ax.set_xticklabels(list_steps_str, rotation='vertical', fontsize=2)
        ax.set_yticklabels(list_keys, rotation='horizontal', fontsize=2)

        title_color = [14 / 255, 122 / 255, 161 / 255]
        plt.title(sim_mode_str + "simulation summary (low value = inactive)", fontsize=8, color=title_color)
        plt.xlabel("step counter", fontsize=6)
        plt.ylabel("component name", fontsize=6)
        plt.colorbar()

        plt.show()
        yd = 5

    else:
        # create discrete colormap
        cmap = colors.ListedColormap(['yellow', 'blue'])  # yellow for [0, 0.5] , blue for [0.5, 1.0]
        bounds = [0, thresh, 1.0]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(data_array, cmap=cmap, norm=norm)

        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.2)

        ax.set_xticks(np.arange(-.5, num_steps, 1))
        ax.set_yticks(np.arange(-.5, num_nodes, 1))

        ax.set_xticklabels(list_steps_str, rotation='vertical', fontsize=2)
        ax.set_yticklabels(list_keys, rotation='horizontal', fontsize=2)

        title_color = [14/255,122/255,161/255]
        plt.title(sim_mode_str + "simulation summary (yellow = inactive)",fontsize=8,color=title_color)
        plt.xlabel("step counter", fontsize=6)
        plt.ylabel("component name", fontsize=6)

        # plt.show()

    fig.savefig(figure_save_fullpath, dpi=300)
#=======================================================================================================================
#=======================================================================================================================
