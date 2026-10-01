import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')

def simulate_trades(n_trades=1000, initial_balance=10000, fee_rate=0.001, risk_per_trade=0.03,
                    stop_median=0.03, stop_sigma=0.0, max_leverage=1.0, scenarios=None):
    """
    Simulate trading outcomes with multiple risk-reward scenarios, trading fees
    and risk-based position sizing with a variable stop distance

    Parameters:
    - n_trades: number of trades to simulate
    - initial_balance: starting account balance
    - fee_rate: trading fee as a decimal (e.g., 0.001 for 0.1%), paid on entry and exit
    - risk_per_trade: fraction of the balance lost when the stop is hit, i.e. 1R (e.g., 0.03 for 3%)
    - stop_median: median stop distance as a fraction of entry price (e.g., 0.03 for 3%)
    - stop_sigma: log-normal spread of the stop distance (0 = fixed stop distance)
    - max_leverage: cap on position size / balance (1.0 = spot, no leverage)
    - scenarios: list of tuples (probability, risk_reward_ratio)

    Position size = balance * risk_per_trade / stop_distance, capped at balance * max_leverage.
    When the cap binds, the actual risk is smaller than risk_per_trade.
    With stop_sigma=0, max_leverage=1 and risk_per_trade == stop_median the whole
    balance is traded every time (the original fixed BASE_RATE model).
    """
    if scenarios is None:
        scenarios = [(0.41, 1.5), (0.06, 2.0), (0.02, 2.5), (0.01, 3.0)]

    # Calculate loss probability dynamically
    total_win_prob = sum(prob for prob, _ in scenarios)
    loss_prob = 1 - total_win_prob
    if not 0 <= loss_prob <= 1 or any(prob < 0 for prob, _ in scenarios):
        raise ValueError(f"Scenario probabilities must be >= 0 and sum to at most 1 (sum = {total_win_prob})")

    # Outcome buckets: [Loss, Win_1, Win_2, ...]; the last threshold is forced to 1.0
    # so floating point rounding can never leave a draw without a bucket
    thresholds = np.cumsum([loss_prob] + [prob for prob, _ in scenarios])
    thresholds[-1] = 1.0
    labels = np.array(['Loss'] + [f'Win_{rrr}x' for _, rrr in scenarios], dtype=object)
    r_multiples = np.array([-1.0] + [rrr for _, rrr in scenarios])

    random_outcomes = np.random.random(n_trades)
    buckets = np.searchsorted(thresholds, random_outcomes, side='right')
    stop_distances = stop_median * np.exp(stop_sigma * np.random.standard_normal(n_trades))

    # Initialize arrays
    start_balances = np.zeros(n_trades)
    position_sizes = np.zeros(n_trades)
    fees_paid = np.zeros(n_trades)
    trade_results = np.zeros(n_trades)
    balances = np.zeros(n_trades)

    balance = initial_balance
    fee_factor = (1 - fee_rate) ** 2
    for i in range(n_trades):
        start_balances[i] = balance

        # Risk-based position sizing, capped by leverage
        position = min(balance * risk_per_trade / stop_distances[i], balance * max_leverage)
        effective_position = position * fee_factor
        fees = position - effective_position

        position_sizes[i] = position
        fees_paid[i] = fees
        trade_results[i] = effective_position * stop_distances[i] * r_multiples[buckets[i]]

        balance = balance - fees + trade_results[i]
        balances[i] = balance

    # Create DataFrame with results
    df = pd.DataFrame({
        'Trade_Number': range(1, n_trades + 1),
        'Start_Balance': start_balances,
        'Stop_Distance': stop_distances,
        'Position_Size': position_sizes,
        'Leverage': position_sizes / start_balances,
        'Fees_Paid': fees_paid,
        'Outcome': labels[buckets],
        'Trade_PnL': trade_results,
        'Balance': balances,
        'Return_Per_Trade': balances / start_balances - 1,  # net of fees
        'Random_Value': random_outcomes  # Added for verification
    })

    return df

def max_drawdown(df, initial_balance):
    """Largest peak-to-trough decline, including the initial balance as a possible peak"""
    balance = pd.concat([pd.Series([initial_balance]), df['Balance']], ignore_index=True)
    return (balance / balance.cummax() - 1).min()

def analyze_results(df, initial_balance, risk_per_trade):
    """Print detailed trading statistics"""
    total_trades = len(df)
    outcome_counts = df['Outcome'].value_counts()
    final_balance = df['Balance'].iloc[-1]
    min_balance = df['Balance'].min()
    total_fees = df['Fees_Paid'].sum()

    print(f"\nTotal Trades: {total_trades}")
    print(f"Risk per Trade (1R): {risk_per_trade:.2%}")
    print(f"Stop Distance: median {df['Stop_Distance'].median():.2%}, "
          f"range {df['Stop_Distance'].min():.2%} - {df['Stop_Distance'].max():.2%}")
    print(f"Leverage: mean {df['Leverage'].mean():.2f}x, max {df['Leverage'].max():.2f}x")
    print(f"Average Fees per Trade: {(df['Fees_Paid'] / (df['Start_Balance'] * risk_per_trade)).mean():.3f}R")
    print("\nOutcome Distribution:")
    for outcome, count in outcome_counts.items():
        print(f"{outcome}: {count} trades ({count/total_trades:.2%})")

    print(f"\nInitial Amount: ${initial_balance:,.2f}")
    print(f"Minimum Balance: ${min_balance:,.2f}")
    print(f"Final Balance: ${final_balance:,.2f}")
    print(f"Total Return: {((final_balance/initial_balance) - 1):.2%}")
    print(f"Total Fees Paid: ${total_fees:,.2f}")
    print(f"Max Drawdown: {max_drawdown(df, initial_balance):.2%}")


# Example usage
#np.random.seed(1337)  # For reproducibility

# Example with different scenarios
scenarios = [
    (0.41, 1.5),  # 41% chance of 1.5R win
    (0.06, 2.0),  # 6% chance of 2R win
    (0.02, 2.5),  # 2% chance of 2.5R win
    (0.01, 3.0)   # 1% chance of 3R win
    # Loss probability will automatically be 50% (1 - 0.50)
]

RISK_PER_TRADE = 0.0333  # 1R as a fraction of the balance
STOP_MEDIAN = 0.0333     # median stop distance from entry price
STOP_SIGMA = 0.5         # 0 = fixed stop distance; 0.5 -> ~68% of stops within 0.6x - 1.65x of the median
MAX_LEVERAGE = 3.0       # 1.0 = spot only
INITIAL_ASSET = 400
N_TRADES = 500
FEE_RATE = 0.00075

sim_params = dict(
    n_trades = N_TRADES,
    initial_balance = INITIAL_ASSET,
    fee_rate = FEE_RATE,
    risk_per_trade = RISK_PER_TRADE,
    stop_median = STOP_MEDIAN,
    stop_sigma = STOP_SIGMA,
    max_leverage = MAX_LEVERAGE,
    scenarios = scenarios
)

results = simulate_trades(**sim_params)

# Analyze and plot results
analyze_results(results, INITIAL_ASSET, RISK_PER_TRADE)



plt.figure(figsize=(12, 6))
plt.plot(results['Trade_Number'], results['Balance'])
plt.title('Account Balance Over Time')
plt.xlabel('Number of Trades')
plt.ylabel('Balance ($)')
plt.grid(True,linestyle='--', alpha=0.2)
plt.show()



# Monte Carlo Experiment
mc_rows = []

for i in range(200):
    mc_run = simulate_trades(**sim_params)
    counts = mc_run.Outcome.value_counts()

    mc_rows.append({
        "Loss": counts.get('Loss', 0),
        "Win": (mc_run.Outcome != 'Loss').sum(),
        "Minimum Balance": mc_run.Balance.min(),
        "Final Balance": mc_run.Balance.iloc[-1],
        "Total Fees": mc_run.Fees_Paid.sum(),
        "Max Drawdown": max_drawdown(mc_run, INITIAL_ASSET)
    })

df2 = pd.DataFrame(mc_rows)

print(df2[["Max Drawdown", "Minimum Balance", "Final Balance"]].describe())

# df2["Max Drawdown"].plot()
# plt.show()
#
# df2["Final Balance"].plot()
# plt.show()
#
# df2["Minimum Balance"].plot()
# plt.show()

for i in range(5):
    print(i)
    print(results.iloc[i])
