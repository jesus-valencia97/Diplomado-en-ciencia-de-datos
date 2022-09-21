
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def formater(str_number):
    try:
        str_number = float(str_number)
        if str_number.is_integer():
            return "{:,.{}f}".format(str_number, 0)
        else:
            return "{:,.{}f}".format(str_number, 1)
    except:
        return str_number

def bar(db,variable,title='',barlabs=None,barlabsrot = 'horizontal',rot=0,top=None,ax=None,scaled = None,format=None,yvisible=False,palette='Wistia_r'):
    
    if top is not None: 
        counts=db[variable].value_counts().head(top)
    else:
        counts=db[variable].value_counts()

    if scaled:
        counts = counts/scaled
    
    if ax is not None:
        g= sns.barplot(x=counts.index,y=counts,palette=palette,ax=ax,edgecolor = 'white')
        ax.set_title(title,fontsize=30,fontweight='bold')
        if barlabs==True:
            if format:
                g.bar_label(g.containers[0],labels=[formater(x) for x in g.containers[0].datavalues],rotation=barlabsrot)
            else:
                g.bar_label(g.containers[0],rotation=barlabsrot)
        else:
            pass
        ax.tick_params(labelrotation=rot)
        ax.set_yticks([])
    else:
        plt.figure(figsize=(10,6))
        
        if yvisible:
            g= sns.barplot(x=[formater(c) for c in counts.index],y=counts,palette=palette,ax=ax,edgecolor = 'white', alpha = 0.5)
            if barlabs==True:
                if format:
                    g.bar_label(g.containers[0],labels=[formater(x) for x in g.containers[0].datavalues],rotation=barlabsrot)
                else:
                    g.bar_label(g.containers[0],rotation=barlabsrot)
            else:
                pass
            plt.xticks(rotation=rot)

            plt.gca().get_yaxis().set_visible(False)
        else:
            g= sns.barplot(x=[formater(c) for c in counts.index],y=counts,palette=palette,ax=ax,edgecolor = 'white',alpha=0.5)
            sns.despine(bottom = True, left = True)
            if barlabs==True:
                if format:
                    g.bar_label(g.containers[0],labels=[formater(x) for x in g.containers[0].datavalues],rotation=barlabsrot)
                else:
                    g.bar_label(g.containers[0],rotation=barlabsrot)
            else:
                pass
            plt.xticks(rotation=rot)
            plt.gca().set_yticks([])

            plt.gca().get_yaxis().set_visible(False)

        plt.suptitle(title,size=30,fontweight='bold') if title else plt.suptitle(variable,size=30,fontweight='bold')

def pie(db,variable,title=None,labels = True,legend = True,threshold = 0):
    counts=db[variable].value_counts(normalize=True)
    lbs = counts.index.values
    n = len(lbs)

    def my_level_list(data,threshold):
        list = []
        for i in range(len(data)):
            if (data[i]*100/np.sum(data)) > threshold : #2%
                list.append(lbs[i])
            else:
                list.append('')
        return list

    def my_autopct(pct):
        return formater(pct) + '%' if pct > threshold else ''

    plt.figure(figsize=(10,10))

    if labels:
        patches, labeltext, pcts = plt.pie(x = counts,labels = my_level_list(counts,threshold),colors = sns.color_palette('pastel',n),autopct=my_autopct,textprops = dict(rotation_mode = 'default', va='center', ha='center'))

        for i,l in enumerate(labeltext):
            l.set_fontsize(250*(1/(4*(i+1))))

        for i,l in enumerate(pcts):
            l.set_fontsize(1.25*l.get_fontsize()*(1-0.05*i))

    else:
        plt.pie(x = counts,colors = sns.color_palette('pastel',len(lbs)))
    
    if legend:
        plt.legend(lbs)

    plt.suptitle(title,size=30,fontweight='bold') if title else plt.suptitle(variable,size=30,fontweight='bold')
    # plt.setp(labels, fontsize=15)



def hist(db, variable,ctitle=None,nbins=10,logx=False,logy=False):

    plt.figure(figsize=(8,8))

    db[variable].plot(kind="hist", logx=logx, logy=logy,bins=nbins, histtype='stepfilled',alpha=0.3, ec="k",edgecolor="skyblue", linewidth=2 ,label='_nolegend_')
    
    # plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(':,.2f'))
    # plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(round(x)))))
    plt.gca().xaxis.set_major_formatter(tkr.FuncFormatter(lambda y,  p: format(int(round(y,0)), ',')))

    # plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('{x:,}'))
    # plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(round(x)))))
    plt.gca().yaxis.set_major_formatter(tkr.FuncFormatter(lambda y,  p: format(int(round(y,0)), ',')))
    
    plt.grid(False)
    plt.axvline(x=db[variable].quantile(0.25),color='blue',alpha=0.25, ymax=0.90,ls='--')
    plt.axvline(x=db[variable].quantile(0.5),color='blue',alpha=0.5, ymax=0.90,ls='--')
    plt.axvline(x=db[variable].quantile(0.75),color='blue',alpha=0.75, ymax=0.90,ls='--')
    plt.legend((r'$q_{0.25}$',r'$q_{0.50}$',r'$q_{0.75}$'))
    plt.suptitle(ctitle,size=20,fontweight='bold',y=0.92) if ctitle else plt.suptitle(variable,size=20,fontweight='bold',y=0.92)

def hist_box(db, variable,ctitle=None,nbins=10,logx=False,logy=False):
    f, (ax_box, ax_hist) = plt.subplots(2,figsize=(12,8), sharex=True, gridspec_kw={"height_ratios": (.20, .80)})
    
    db[variable].plot(kind="hist", ax=ax_hist,logx=logx, logy=logy,bins=nbins, histtype='stepfilled',alpha=0.3, ec="k",edgecolor="skyblue", linewidth=2 ,label='_nolegend_')
    db[variable].plot(kind="box", ax=ax_box,notch=True, logx=logx,logy=logy ,vert=False,label='',color='navy',flierprops = dict(marker='.', markerfacecolor='skyblue', markersize=7,linestyle = '',markeredgecolor='black',markeredgewidth=0.1))

    # plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(':,.2f'))
    # plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(round(x)))))
    plt.gca().xaxis.set_major_formatter(tkr.FuncFormatter(lambda y,  p: format(int(round(y,0)), ',')))

    # plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('{x:,}'))
    # plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(round(x)))))
    plt.gca().yaxis.set_major_formatter(tkr.FuncFormatter(lambda y,  p: format(int(round(y,0)), ',')))
    
    plt.grid(False)
    plt.axvline(x=db[variable].quantile(0.25),color='blue',alpha=0.25, ymax=0.90,ls='--')
    plt.axvline(x=db[variable].quantile(0.5),color='blue',alpha=0.5, ymax=0.90,ls='--')
    plt.axvline(x=db[variable].quantile(0.75),color='blue',alpha=0.75, ymax=0.90,ls='--')
    plt.legend((r'$q_{0.25}$',r'$q_{0.50}$',r'$q_{0.75}$'))
    f.suptitle(ctitle,size=20,fontweight='bold',y=0.92) if ctitle else f.suptitle(variable,size=20,fontweight='bold',y=0.92)

def hist_by_var(db,var,by, normalize = False):
    if var == by:
            pass
    else:
        if normalize:
                try:
                    aux = pd.DataFrame(pd.cut(db[var],bins = 30)).set_axis([var],axis=1)
                    aux[by] = db[by]
                    aux = aux.value_counts().reset_index().pivot_table(index=var,columns = by,values=0,fill_value=0)
                    aux = aux.T
                    aux.columns = aux.columns.map(lambda x: x.left)
                    aux['TOTAL'] = aux.apply(sum,axis=1)
                    aux = aux.apply(lambda x : x / sum(x), axis=0)
                    ax = aux.T.plot(kind='bar',stacked=True,width=0.9)
                    plt.locator_params(axis='x', nbins=6)
                    plt.xticks(rotation=0)

                    for c in ax.containers:

                        # Optional: if the segment is small or 0, customize the labels
                        labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
                        
                        # remove the labels parameter if it's not needed for customized labels
                        ax.bar_label(c, labels=labels, label_type='center')

                    plt.show()
                except Exception as e:
                    print(e)
                    aux = db[[var]]
                    aux[by] = db[by]
                    aux = aux.value_counts().reset_index().pivot_table(index=var,columns = by,values=0,fill_value=0)
                    aux = aux.T
                    # aux.columns = aux.columns.map(lambda x: x.left)
                    aux['TOTAL'] = aux.apply(sum,axis=1)
                    aux = aux.apply(lambda x : x / sum(x), axis=0)
                    ax = aux.T.plot(kind='bar',stacked=True,width=0.9)
                    # plt.locator_params(axis='x', nbins=6)
                    # plt.xticks(rotation=0)
                    

                    for c in ax.containers:

                        # Optional: if the segment is small or 0, customize the labels
                        labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
                        
                        # remove the labels parameter if it's not needed for customized labels
                        ax.bar_label(c, labels=labels, label_type='center')

                    plt.show()
        else:
                try:
                    aux = pd.DataFrame(pd.cut(db[var],bins = 30)).set_axis([var],axis=1)
                    aux[by] = db[by]
                    aux = aux.value_counts().reset_index().pivot_table(index=var,columns = by,values=0,fill_value=0)
                    aux = aux.T
                    aux.columns = aux.columns.map(lambda x: x.left)
                    ax = aux.T.plot(kind='bar',stacked=True,width=1)
                    plt.locator_params(axis='x', nbins=6)
                    plt.xticks(rotation=0)

                    for c in ax.containers:

                        # Optional: if the segment is small or 0, customize the labels
                        labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
                        
                        # remove the labels parameter if it's not needed for customized labels
                        ax.bar_label(c, labels=labels, label_type='center')

                    plt.show()

                    
                except Exception as e:
                    print(e)
                    aux = db[[var]]
                    aux[by] = db[by]
                    aux = aux.value_counts().reset_index().pivot_table(index=var,columns = by,values=0,fill_value=0)
                    aux = aux.T
                    # aux.columns = aux.columns.map(lambda x: x.left)
                    ax = aux.T.plot(kind='bar',stacked=True,width=1)
                    # plt.locator_params(axis='x', nbins=6)
                    # plt.xticks(rotation=0)

                    for c in ax.containers:

                        # Optional: if the segment is small or 0, customize the labels
                        labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
                        
                        # remove the labels parameter if it's not needed for customized labels
                        ax.bar_label(c, labels=labels, label_type='center')

                    plt.show()