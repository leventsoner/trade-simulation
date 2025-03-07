import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_trades(n_trades=1000, initial_trade_amount=10000, fee_rate=0.001, base_rate=0.03, scenarios=None):
    """
    Simulate trading outcomes with multiple risk-reward scenarios and trading fees
    
    Parameters:
    - n_trades: number of trades to simulate
    - initial_trade_amount: starting amount for first trade
    - fee_rate: trading fee as a decimal (e.g., 0.001 for 0.1%)
    - base_rate: base rate for 1R (e.g., 0.03 for 3%)
    - scenarios: list of tuples (probability, risk_reward_ratio)
    """
    if scenarios is None:
        scenarios = [(0.41, 1.5), (0.06, 2.0), (0.02, 2.5), (0.01, 3.0)]
    
    # Calculate loss probability dynamically
    total_win_prob = sum(prob for prob, _ in scenarios)
    loss_prob = 1 - total_win_prob
    win_start = loss_prob  # Start of win probabilities
    
    # Initialize arrays
    random_outcomes = np.random.random(n_trades)
    trade_results = np.zeros(n_trades)
    outcome_types = np.full(n_trades, 'Loss')
    fees_paid = np.zeros(n_trades)
    effective_amounts = np.zeros(n_trades)
    trade_amounts = np.zeros(n_trades)
    
    # Set initial trade amount
    trade_amounts[0] = initial_trade_amount
    
    # Calculate cumulative probabilities for win scenarios
    cum_probs = []
    current_threshold = win_start
    for prob, rrr in scenarios:
        current_threshold += prob
        cum_probs.append((current_threshold, rrr))
    
    for i in range(n_trades):
        current_trade_amount = trade_amounts[i]
        
        # Calculate effective trade amount and fees for this trade
        effective_amount = current_trade_amount * ((1-fee_rate)**2)
        fees = current_trade_amount - effective_amount
        
        effective_amounts[i] = effective_amount
        fees_paid[i] = fees
        
        # Determine trade outcome
        if random_outcomes[i] < loss_prob:  # Dynamic loss probability
            trade_results[i] = -effective_amount * base_rate
            outcome_types[i] = 'Loss'
        else:
            # Check which win scenario applies
            for threshold, rrr in cum_probs:
                if random_outcomes[i] < threshold:
                    profit_rate = base_rate * rrr
                    trade_results[i] = effective_amount * profit_rate
                    outcome_types[i] = f'Win_{rrr}x'
                    break
        
        # Calculate next trade amount (if not the last trade)
        if i < n_trades - 1:
            trade_amounts[i + 1] = effective_amount + trade_results[i]
    
    # Create DataFrame with results
    df = pd.DataFrame({
        'Trade_Number': range(1, n_trades + 1),
        'Trade_Amount': trade_amounts,
        'Effective_Amount': effective_amounts,
        'Fees_Paid': fees_paid,
        'Outcome': outcome_types,
        'Trade_PnL': trade_results,
        'Balance': effective_amounts + trade_results,
        'Return_Per_Trade': trade_results / trade_amounts,
        'Random_Value': random_outcomes  # Added for verification
    })
    
    return df

def analyze_results(df, initial_trade_amount, base_rate):
    """Print detailed trading statistics"""
    total_trades = len(df)
    outcome_counts = df['Outcome'].value_counts()
    final_balance = df['Balance'].iloc[-1]
    min_balance = df['Balance'].min()
    total_fees = df['Fees_Paid'].sum()
    max_drawdown = (df['Balance'].div(df['Balance'].cummax()) - 1).min()
    
    print(f"\nTotal Trades: {total_trades}")
    print(f"Base Rate (1R): {base_rate:.1%}")
    print("\nOutcome Distribution:")
    for outcome, count in outcome_counts.items():
        print(f"{outcome}: {count} trades ({count/total_trades:.2%})")
    
    print(f"\nInitial Amount: ${initial_trade_amount:,.2f}")
    print(f"Minimum Balance: ${min_balance:,.2f}")
    print(f"Final Balance: ${final_balance:,.2f}")
    print(f"Total Return: {((final_balance/initial_trade_amount) - 1):.2%}")
    print(f"Total Fees Paid: ${total_fees:,.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

# Example usage
np.random.seed(1337)  # For reproducibility

# Example with different scenarios
scenarios = [
    (0.30, 1.5),  # 30% chance of 1.5R win
    (0.10, 2.0),  # 10% chance of 2R win
    (0.05, 2.5),  # 5% chance of 2.5R win
    (0.05, 3.0)   # 5% chance of 3R win
    # Loss probability will automatically be 50% (1 - 0.50)
]

results = simulate_trades(
    n_trades=1000,
    initial_trade_amount=10000,
    fee_rate=0.001,
    base_rate=0.03,
    scenarios=scenarios
)

# Analyze and plot results
analyze_results(results, 10000, 0.03)

plt.figure(figsize=(12, 6))
plt.plot(results['Trade_Number'], results['Balance'])
plt.title('Account Balance Over Time')
plt.xlabel('Number of Trades')
plt.ylabel('Balance ($)')
plt.grid(True)
plt.show()