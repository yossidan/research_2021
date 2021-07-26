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
import sys
sys.path.append('./Source/')
from BooleanModel import BooleanModel
from Utilities import *

#=======================================================================================================================
if __name__ == '__main__':
#=======================================================================================================================

    path_to_sim_inifile = "./_inputs/sim_1/sim_1.ini"

    try:
        model = BooleanModel(path_to_sim_inifile)
    except:
        print("\n\n---> stopping simulation due to some exception during model initialization...")
        sys.exit(18)

    sim_mode = model.get_sim_mode()

    final_results_all_steps = model.run_model_simulation()
    #
    # final_results_all_steps = a "table of size N_NODES x N_STEPS.
    #
    # print(final_results_all_steps)

    outputs_path = model.get_outputs_path()

    write_data(final_results_all_steps, outputs_path)

    title_aux_str = sim_mode + ' '
    my_grid_plot(final_results_all_steps, title_aux_str, 0.5, outputs_path)

    # plot_result(final_results_all_steps, model.SIM_PARAMS['plot_nodes'], marker=False)
#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================
