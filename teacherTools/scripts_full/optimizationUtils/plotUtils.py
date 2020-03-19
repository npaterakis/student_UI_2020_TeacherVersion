import matplotlib.pyplot as plt
import pandas

def plot_bid(df):
    plt.title('Bid quantity')
    plt.plot(df, color='black', marker='.')
    plt.ylabel('Bid quantity (MWh)')
    plt.xlabel('Period (h)')
    plt.xticks(rotation='vertical')
    plt.grid()
    plt.show()

def plot_profit_distribution(df, df2):

    plt.title('Profit distribution with parameters: alpha '+str(df2.loc['alpha'])+' and beta '+str(df2.loc['beta']))
    plt.scatter(df.loc[:,'profit'], df.loc[:,'cumprob'], marker='.')
    plt.scatter(df2.loc['CVaR'], y=0, marker="^", label= 'CVaR')
    plt.scatter(df2.loc['expected_profit'], y=0, marker="^", label='Expected profit')
    plt.ylabel('Probability')
    plt.xlabel('Profit (Euros)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_hourly_imbalance_dists(df1, df2, probs=None):
    plt.title('Hourly imbalance volume statistics')
    plt.plot(df1.max(axis=1), linewidth=0, marker='^', label = 'Max')
    plt.plot(df1.min(axis=1), linewidth=0, marker='v', label = 'Min')
    plt.plot(df2, color='black', marker='*', linestyle='dashed', label = 'Expected')
    plt.ylabel('Imbalance (MWh)')
    plt.xlabel('Period (h)')
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.grid()
    plt.show()