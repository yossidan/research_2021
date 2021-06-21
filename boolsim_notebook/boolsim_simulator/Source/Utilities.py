import sys
import os

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

        #print(data_array.shape)

        axis_steps = np.arange(-.5, num_steps, 1)
        axis_nodes = np.arange(-.5, num_nodes, 1)

        # handle case where dims of axis_steps, axis_nodes don't match the dim of data_array.
        # because on some python versions, the axis_steps, axis_nodes are 1 unit longer.
        nsteps = len(axis_steps)
        nnodes = len(axis_nodes)
        diff_steps = num_steps - nsteps
        diff_nodes = num_nodes - nnodes
        if diff_steps < 0:
            axis_steps = axis_steps[:diff_steps]
        if diff_nodes < 0:
            axis_nodes = axis_nodes[:diff_nodes]

        ax.set_xticks(axis_steps)
        ax.set_yticks(axis_nodes)
        # ax.set_xticks(np.arange(-.5, num_steps, 1))
        # ax.set_yticks(np.arange(-.5, num_nodes, 1))

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
#=======================================================================================================================
