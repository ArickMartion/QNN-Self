
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output




def plot_bar3d(X,Y,Z,width=0.3,depth=0.3,colors=None,shade=True,xlabel="x",ylabel="y",zlabel="z",title=None,figsize=(6,6),edgecolor="black",
               bottom=0,zlim=None,xticks=None,yticks=None,xticklabels=None,yticklabels=None,save=False,path=None,show=False):
    """ Plot a 3D bar chart
    """

    # Plot settings
    fig=plt.figure(figsize=figsize)
    ax=plt.axes(projection="3d") 

    xx,yy=np.meshgrid(X,Y)
    X_,Y_=xx.ravel(),yy.ravel() 

    # Start plotting
    if colors is None:
        colors=["royalblue"]*len(Z)
    
    ax.bar3d(X_,Y_,bottom,width,depth,Z,color=colors,edgecolor=edgecolor,shade=shade)
    #ax.bar3d(X_,Y_,bottom,width,depth,Z)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    if zlim is not None:
        ax.set_zlim(zlim) 
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    if title is not None:
        plt.title(title)
    
    if save==True:
        plt.savefig(path, dpi=500)
    if show==True:
        plt.show()

def plot_bar3d_subplot(X,Y,Z_ls,shape=(3,3),width=0.3,depth=0.3,colors=None,shade=True,xlabel="x",ylabel="y",zlabel="z",title=None,
                       figsize=(6,6),edgecolor="black",bottom=0,zlim=None,xticks=None,yticks=None,xticklabels=None,
                       yticklabels=None,save=False,path=None,show=False):
    """Plot a 3D bar chart
    """
    
    # Create a figure with a 2x2 subplot grid
    fig, axs = plt.subplots(shape[0], shape[1], subplot_kw={'projection': '3d'}, figsize=figsize)

    # Plot subplots
    for i, ax in enumerate(axs.flat):
        if i>=len(Z_ls):
            continue
        
        xx,yy=np.meshgrid(X,Y)
        X_,Y_=xx.ravel(),yy.ravel() 

        # Begin plotting
        if colors is None:
            colors=["royalblue"]*len(Z_ls[i])

        ax.bar3d(X_,Y_,bottom,width,depth,Z_ls[i],color=colors,edgecolor=edgecolor,shade=shade)
        #ax.bar3d(X_,Y_,bottom,width,depth,Z)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        if zlim is not None:
            ax.set_zlim(zlim)
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)
        if title is not None:
            plt.title(title)
    
    if save==True:
        plt.savefig(path, dpi=500)
    if show==True:
        plt.show()
        
# In[3]:

def plot(x,y,xlabel="x",ylabel="y",title="x-y",labels="x-y",colors=["b"],sizes=[2],markers=["o"],markersizes=[2],
         figsize=(6,4),save=False,path=None,clear=False,show=True,legend_loc=None,plot_type="plot"):
    """功能：绘图，若输入多个，则在同1个fig上画多个图，此时 y、plot_type 应为同大小的列表"""

    fig=plt.figure(figsize=figsize)

    # If it's a list, loop through to plot
    if type(y)==list:
        if len(y)!=len(labels): 
            labels=[k for k in range(len(y))]
        if len(y)!=len(colors):
            colors=[None]*len(y)
        if len(y)!=len(sizes):
            sizes=[1]*len(y)
        if len(y)!=len(markers):
            markers=["o"]*len(y)
        if len(y)!=len(markersizes):
            markersizes=[1]*len(y)
         
        for k in range(0,len(y)):
            if plot_type=="plot":
                plt.plot(x,y[k],color=colors[k],linewidth=sizes[k],label=labels[k],marker=markers[k],markersize=markersizes[k]) #   
            elif plot_type=="loglog":
                plt.loglog(x,y[k],color=colors[k],linewidth=sizes[k],label=labels[k],marker=markers[k],markersize=markersizes[k]) #  
            elif plot_type=="scatter":
                plt.scatter(x,y[k],color=colors[k],label=labels[k],marker=markers[k],markersize=markersizes[k]) #
                
    else:
        plt.plot(x,y[k],label=labels,marker=markers) #  
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=legend_loc)
    
    if save==True:
        plt.savefig(path, dpi=500)
    if show==True:
        plt.show()    
    if clear==True:
        plt.cla()


def middle_plot(x,y,xlabel="x",ylabel="y",title="x-y",labels="x-y",colors="b",sizes="2",markers="o",
         figsize=(6,4),save=False,path=None,clear=False,show=True,legend_loc=None,plot_type="plot"):
    """
    Function: Plotting. If multiple inputs are provided, plot them on the same figure. 
    In this case, `y` and `plot_type` should be lists of the same length.
    """

    # If it's a list, loop through to plot
    if type(y)==list:
        if len(y)!=len(labels): 
            labels=[k for k in range(len(y))]
        if len(y)!=len(colors):
            colors=[None]*len(y)
        if len(y)!=len(sizes):
            sizes=[2]*len(y)
        if len(y)!=len(markers):
            markers=["o"]*len(y)
         
        for k in range(0,len(y)):
            if plot_type=="plot":
                plt.plot(x,y[k],color=colors[k],linewidth=sizes[k],label=labels[k],marker=markers[k]) #
            elif plot_type=="loglog":
                plt.loglog(x,y[k],color=colors[k],linewidth=sizes[k],label=labels[k],marker=markers[k]) #
            elif plot_type=="scater":
                plt.loglog(x,y[k],color=colors[k],label=labels[k],marker=markers[k]) #
                
    else:
        plt.plot(x,y[k],label=labels,marker=markers) #
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=legend_loc)
    
    if save==True:
        plt.savefig(path, dpi=500)
    if show==True:
        plt.show()    
    if clear==True:
        plt.cla()


def plot_training_progress(data,xlabel="Iteration",ylabel="Loss",title="Loss",label="generator_loss",
                           color="royalblue",figsize=(6,4),save=False,path=None,show=True):

    fig,ax1=plt.subplots(1,1,figsize=figsize)
    
    ax1.set_title(title)
    
    num_datas=len(data)
    for k in range(num_datas):
        ax1.plot(data[k],label=label[k],color=color[k])
    ax1.legend(loc="best")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.grid()
    
    if save==True:
        plt.savefig(path,dpi=500)
    if show==True:
        plt.show()
        
        
def plot_classified_points(data=[], colors=[["red","blue"]], xlim=[0,np.pi],ylim=[0,np.pi],
                           title='Classified Points', figsize=(4,4), alphas=[1],      
                           legend=False,ponts_sizes=[None], classes=2, subplots=None, save_path=None, 
                          fonts = [None,None,None],
                           tick_labels = [None, None],
                           labels = [r"$x_1$",r"$x_2$"],
                           edgecolors = [["red","blue"],["red","blue"]],
                           linewidths = [1,1],
                           dpi=200
                          ):
    """
    Plot 2D data points, using different colors based on the values of `y`.
    
    Parameters:
        x1 (array-like): Data for the first dimension.
        x2 (array-like): Data for the second dimension.
        y (array-like): Class labels, with y > 0 in red and y <= 0 in blue.
"""
    
    fig,ax1=plt.subplots(figsize=figsize)
    
    if subplots is None:    
        num_plots=1
    elif subplots is not None:
        num_plots=subplots[0]*subplots[1]
        
    for n in range(0,num_plots):
        
        if subplots is not None:
            plt.subplot(subplots[0],subplots[1],n+1)
            N=len(data[n])
        else:
            N=len(data)
            
        #同时绘制多个
        for k in range(N):
            if subplots is not None:
                x=data[n][k][0]
                y=data[n][k][1]
            else:
                x=data[k][0]
                y=data[k][1]
                
            x1,x2=x[:,0],x[:,1]
            y = np.array(y)

            for c in range(classes):
                points=(y==c)
                plt.scatter(x1[points], x2[points], color=colors[k][c], label=f'class{c}', alpha=alphas[k], s=ponts_sizes[k],
                           edgecolors=edgecolors[k][c], linewidths = linewidths[k]
                           )
            

        plt.xlim(xlim)
        plt.ylim(ylim)
        
        if tick_labels[0] is not None:
            ax1.set_xticks(tick_labels[0])
        if tick_labels[1] is not None:
            ax1.set_yticks(tick_labels[1])

        # Add legend and title
        plt.xlabel(labels[0], fontdict=fonts[0])
        plt.ylabel(labels[1], fontdict=fonts[0])
        
        #plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)  # Horizontal line
        #plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)  # Vertical line
        
        #ax1.tick_params(labelsize=fonts[1]["size"])
        labels=ax1.get_xticklabels()+ax1.get_yticklabels()
        [label.set_font(fonts[1]) for label in labels]
        
        
        plt.title(title, fontdict=fonts[2])

        if legend==True:
            plt.legend()
        plt.grid(alpha=0.3)
        
        #plt.tight_layout()
        
    if save_path is not None:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.show()
    
def plot_mean_std(data_dict,title="Training Curves with Mean ± Std"):
    """
    Plot the mean and standard deviation shadow of multiple curves.
    :param data_dict: dict, {key: [list_of_runs], ...}
                      Each key corresponds to a model or method,
                      list_of_runs contains multiple training curves, 
                      each curve being a list or array.
    """
    plt.figure(figsize=(8, 6))
    
    for key, runs in data_dict.items():
        # Convert to NumPy array for easier column-wise computation
        arr = np.array(runs)  # shape: (num_runs, num_iterations)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        
        x = np.arange(len(mean))

        plt.plot(x, mean, label=key)

        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()