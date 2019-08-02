import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib import style
from datetime import datetime

# finance module is no longer part of matplotlib
# see: https://github.com/matplotlib/mpl_finance
from mpl_finance import candlestick_ochl as candlestick

style.use('dark_background')

VOLUME_CHART_HEIGHT = 0.33

UP_COLOR = '#27A59A'
DOWN_COLOR = '#EF534F'
UP_TEXT_COLOR = '#73D3CC'
DOWN_TEXT_COLOR = '#DC2C27'


def date2num(date):
    converter = mdates.strpdate2num('%YYYY-%mm-%dd')
    return converter(date)


class StockTradingGraph:
    """A stock trading visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, df, title=None):
        self.df = df
        self.net_worths = np.zeros(len(df['DayOfYear']))
        self.rewards = np.zeros(len(df['DayOfYear']))
        # Format dates as timestamps, necessary for candlestick graph
        self.dates = self.df['DayOfYear'].values + ':' + self.df['Time'].values
        # Create a figure on screen and set the title
        fig = plt.figure()
        fig.suptitle(title)

        # Create top subplot for net worth axis
        self.net_worth_ax = plt.subplot2grid(
            (6, 1), (0, 0), rowspan=2, colspan=1)
        # Create a new axis for reward axis
        self.reward_ax = plt.subplot2grid(
            (6, 1), (2, 0), rowspan=2, colspan=1, sharex=self.net_worth_ax)

        # Create bottom subplot for price axis
        # self.price_ax = plt.subplot2grid(
        #     (6, 1), (2, 0), rowspan=2, colspan=1, sharex=self.net_worth_ax)

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0)

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def _render_net_worth(self, current_step, net_worth, step_range, dates):
        # Clear the frame rendered last step
        self.net_worth_ax.clear()

        # Plot net worths
        self.net_worth_ax.plot_date(
            dates, self.net_worths[step_range], '-', label='Net Worth')

        # Show legend, which uses the label we defined for the plot above
        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = pd.to_datetime(str(self.df["DayOfYear"][current_step]), format='%Y%m%d')
        last_net_worth = self.net_worths[current_step]

        # Annotate the current net worth on the net worth graph
        self.net_worth_ax.annotate('{0:.2f}'.format(net_worth), (last_date, last_net_worth),
                                   xytext=(last_date, last_net_worth),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        # Add space above and below min/max net worth
        self.net_worth_ax.set_ylim(
            min(self.net_worths[np.nonzero(self.net_worths)]) / 1.25, max(self.net_worths) * 1.25)

    def _render_reward(self, current_step, reward, step_range, dates):
        # Clear the frame rendered last step
        self.reward_ax.clear()

        # Plot net worths
        self.reward_ax.plot_date(
            dates, self.rewards[step_range], '-', label='Reward')

        # Show legend, which uses the label we defined for the plot above
        self.reward_ax.legend()
        legend = self.reward_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = pd.to_datetime(str(self.df["DayOfYear"][current_step]), format='%Y%m%d')
        last_reward = self.rewards[current_step]

        # Annotate the current net worth on the net worth graph
        self.reward_ax.annotate('{0:.2f}'.format(reward), (last_date, last_reward),
                                   xytext=(last_date, last_reward),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        # Add space above and below min/max net worth
        self.reward_ax.set_ylim(
            min(self.rewards[np.nonzero(self.rewards)]) / 1.25, max(self.rewards) * 1.25)

    def _render_price(self, current_step, net_worth, step_range, dates):
        self.price_ax.clear()

        # Format data for OHCL candlestick graph
        candlesticks = zip(dates,
                           self.df['Open'].values[step_range], self.df['Close'].values[step_range],
                           self.df['High'].values[step_range], self.df['Low'].values[step_range])

        # Plot price using candlestick graph from mpl_finance
        candlestick(self.price_ax, candlesticks, width=1,
                    colorup=UP_COLOR, colordown=DOWN_COLOR)

        last_date = pd.to_datetime(str(self.df["DayOfYear"][current_step]), format='%Y%m%d')
        last_close = self.df['Close'].values[current_step]
        last_high = self.df['High'].values[current_step]

        # Print the current price to the price axis
        self.price_ax.annotate('{0:.2f}'.format(last_close), (last_date, last_close),
                               xytext=(last_date, last_high),
                               bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")

        # Shift price axis up to give volume chart space
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0])
                               * VOLUME_CHART_HEIGHT, ylim[1])

    def _render_trades(self, current_step, trades, step_range):
        for trade in trades:
            if trade['step'] in step_range:
                date = date2num(self.df['DayOfYear'].values[trade['step']])
                high = self.df['High'].values[trade['step']]
                low = self.df['Low'].values[trade['step']]

                if trade['type'] == 'buy':
                    high_low = low
                    color = UP_TEXT_COLOR
                else:
                    high_low = high
                    color = DOWN_TEXT_COLOR

                total = '{0:.2f}'.format(trade['total'])

                # Print the current price to the price axis
                self.price_ax.annotate(f'${total}', (date, high_low),
                                       xytext=(date, high_low),
                                       color=color,
                                       fontsize=8,
                                       arrowprops=(dict(color=color)))

    def render_networth(self, net_worth, dates, step_range, current_step):
        self.net_worth_ax.clear()

        # Plot net worths
        self.net_worth_ax.plot_date(
            dates, self.net_worths[step_range], '-', label='Net Worth')

        # Show legend, which uses the label we defined for the plot above
        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = self.dates[current_step]
        last_net_worth = self.net_worths[current_step]

        # Annotate the current net worth on the net worth graph
        self.net_worth_ax.annotate('{0:.2f}'.format(net_worth), (last_date, last_net_worth),
                                   xytext=(last_date, last_net_worth),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

    def render_reward(self, reward, dates, step_range, current_step):
        # Clear the frame rendered last step
        self.reward_ax.clear()

        # Plot net worths
        self.reward_ax.plot_date(
            dates, self.rewards[step_range], '-', label='Reward')

        # Show legend, which uses the label we defined for the plot above
        self.reward_ax.legend()
        legend = self.reward_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = self.dates[current_step]
        last_reward = self.rewards[current_step]

        # Annotate the current net worth on the net worth graph
        self.reward_ax.annotate('{0:.2f}'.format(reward), (last_date, last_reward),
                                xytext=(last_date, last_reward),
                                bbox=dict(boxstyle='round',
                                          fc='w', ec='k', lw=1),
                                color="black",
                                fontsize="small")

    def render(self, current_step, net_worth, reward, window_size=40):
        self.net_worths[current_step] = net_worth
        self.rewards[current_step] = reward

        window_start = max(current_step - window_size, 0)
        step_range = range(window_start, current_step + 1)

        dates = self.dates[step_range]

        self.render_networth(net_worth, dates, step_range, current_step)
        self.render_reward(reward, dates, step_range, current_step)


        #
        # self._render_net_worth(current_step, net_worth, step_range, dates)
        # self._render_reward(current_step, reward, step_range, dates)
        # # self._render_price(current_step, net_worth, dates, step_range)
        # # self._render_trades(current_step, trades, step_range)
        #
        # # Format the date ticks to be more easily read
        self.reward_ax.set_xticklabels(self.df['DayOfYear'].values[step_range], rotation=45,
                                      horizontalalignment='right')

        self.reward_ax.set_xticks(self.reward_ax.get_xticks()[::2])

        #
        # # Hide duplicate net worth date labels
        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)
        #
        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()
