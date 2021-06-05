
import pulp
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import base64

import fastprogress
import sys
sys.path.append("..")
from fpl_opt.transfers import TransferOptimiser


header=st.beta_container()
dataset=st.beta_container()
features=st.beta_container()
modelTraining=st.beta_container()

@st.cache
def get_data(filename):
	df = pd.read_csv(filename)
	return df



with header:
    from PIL import Image
    img = Image.open("pl_icon.png")
      
    # display image using streamlit
    # width is used to set the width of an image
    st.image(img, width=100)
    st.title('Final Year Project')
    st.write('THIS PROJECT USES LINEAR PROGRAMMING ALGORITHM TO FIND AN OPTIMIZED TEAM TAKING INTO ACCOUNT ALL THE FPL RESTRICTIONS')

# random fake data for costs and values
costs = np.random.uniform(low=5, high=20, size=100)
values = costs * np.random.uniform(low=0.9, high=1.1, size=100)

model = pulp.LpProblem("Constrained value maximisation", pulp.LpMaximize)
decisions = [pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
             for i in range(100)]

# PuLP has a slightly weird syntax, but it's great. This is how to add the objective function:
model += sum(decisions[i] * values[i] for i in range(100)), "Objective"

# and here are the constraints
model += sum(decisions[i] * costs[i] for i in range(100)) <= 100  # total cost
model += sum(decisions) <= 10  # total items

model.solve()


expected_scores = np.random.uniform(low=5, high=20, size=100)
prices = expected_scores * np.random.uniform(low=0.9, high=1.1, size=100)
positions = np.random.randint(1, 5, size=100)
clubs = np.random.randint(0, 20, size=100)

df = get_data('players_raw.csv')

df.sort_values(by=['element_type'])

expected_scores = df["total_points"]  # total points from last season
prices = df["now_cost"] / 10
positions = df["element_type"]
clubs = df["team_code"]
# so we can read the results
fname = df["first_name"]
lname = df["second_name"]
pic = df["code"]



def select_team(expected_scores, prices, positions, clubs, total_budget=100, sub_factor=0.2):
    num_players = len(expected_scores)
    model = pulp.LpProblem("Constrained value maximisation", pulp.LpMaximize)
    decisions = [
        pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    captain_decisions = [
        pulp.LpVariable("y{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    sub_decisions = [
        pulp.LpVariable("z{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]


    # objective function:
    model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * expected_scores[i]
                 for i in range(num_players)), "Objective"

    # cost constraint
    model += sum((decisions[i] + sub_decisions[i]) * prices[i] for i in range(num_players)) <= total_budget  # total cost

    # position constraints
    # 1 starting goalkeeper
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 1) == 1
    # 2 total goalkeepers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 1) == 2

    # 3-5 starting defenders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) <= 5
    # 5 total defenders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 2) == 5

    # 3-5 starting midfielders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) <= 5
    # 5 total midfielders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 3) == 5

    # 1-3 starting attackers
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) >= 1
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) <= 3
    # 3 total attackers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 4) == 3

    # club constraint
    for club_id in np.unique(clubs):
        model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if clubs[i] == club_id) <= 3  # max 3 players

    model += sum(decisions) == 11  # total team size
    model += sum(captain_decisions) == 1  # 1 captain
    
    for i in range(num_players):  
        model += (decisions[i] - captain_decisions[i]) >= 0  # captain must also be on team
        model += (decisions[i] + sub_decisions[i]) <= 1  # subs must not be on team

    model.solve()
    print("Total expected score = {}".format(model.objective.value()))

    return decisions, captain_decisions, sub_decisions


decisions, captain_decisions, sub_decisions = select_team(expected_scores.values, prices.values,
                                                          positions.values, clubs.values,
                                                          sub_factor=0.2)
# print results
p=0
if st.button('CLICK TO GENERATE AN OPTIMIZED TEAM'):
    st.markdown("**PLAYING XI:**")
    st.write("GOALKEEPER:")
    for i in range(df.shape[0]):
        if decisions[i].value() != 0:
            if positions[i] == 1:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.write("DEFENDERS:")
    for i in range(df.shape[0]):
        if decisions[i].value() != 0:
            if positions[i] == 2:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.write("MIDFIELDERS:")
    for i in range(df.shape[0]):
        if decisions[i].value() != 0:
            if positions[i] == 3:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.write("FORWARDS:")
    for i in range(df.shape[0]):
        if decisions[i].value() != 0:
            if positions[i] == 4:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.markdown("**SUBSTITUTES:**")

    st.write("GOALKEEPER:")
    for i in range(df.shape[0]):
        if sub_decisions[i].value() == 1:

            if positions[i] == 1:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.write("DEFENDERS:")
    for i in range(df.shape[0]):
        if sub_decisions[i].value() == 1:

            if positions[i] == 2:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.write("MIDFIELDERS:")
    for i in range(df.shape[0]):
        if sub_decisions[i].value() == 1:

            if positions[i] == 3:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]

    st.write("FORWARDS:")
    for i in range(df.shape[0]):
        if sub_decisions[i].value() == 1:

            if positions[i] == 4:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{}{} {} | Points = {}, Price = {}".format(" [Captain] " if captain_decisions[i].value() == 1 else "", fname[i], lname[i], expected_scores[i], prices[i]))
                p=p+prices[i]


    st.write("Total Cost:")
    st.write("{}".format(p))


num_players = 100
current_team_indices = np.random.randint(0, num_players, size=11)  # placeholder
clubs = np.random.randint(0, 20, size=100)  # placeholder
positions = np.random.randint(1, 5, size=100)  # placeholder
expected_scores = np.random.uniform(0, 10, size=100)  # placeholder

#current_sub_indices = np.random.randint(0, num_players, size=4)  # placeholder
#current_captain_indices = current_team_indices[0]  # placeholder

# convert to binary representation
current_team_decisions = np.zeros(num_players) 
current_team_decisions[current_team_indices] = 1
# convert to binary representation
#current_sub_decisions = np.zeros(num_players) 
#current_sub_decisions[current_sub_indices] = 1
# convert to binary representation
#current_captain_decisions = np.zeros(num_players) 
#current_captain_decisions[current_captain_indices] = 1

model = pulp.LpProblem("Transfer optimisation", pulp.LpMaximize)

transfer_in_decisions = [
    pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
    for i in range(num_players)
]
transfer_out_decisions = [
    pulp.LpVariable("y{}".format(i), lowBound=0, upBound=1, cat='Integer')
    for i in range(num_players)
]

next_week_team = [
    current_team_decisions[i] + transfer_in_decisions[i] - transfer_out_decisions[i]
    for i in range(num_players)
]

for i in range(num_players):
    model += next_week_team[i] <= 1
    model += next_week_team[i] >= 0
    model += (transfer_in_decisions[i] + transfer_out_decisions[i]) <= 1
    
# formation constraints
# 1 starting goalkeeper
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 1) == 1

# 3-5 starting defenders
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 2) >= 3
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 2) <= 5

# 3-5 starting midfielders
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 3) >= 3
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 3) <= 5

# 1-3 starting attackers
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 4) >= 1
model += sum(next_week_team[i] for i in range(num_players) if positions[i] == 4) <= 3

# club constraint
for club_id in np.unique(clubs):
    model += sum(next_week_team[i] for i in range(num_players) if clubs[i] == club_id) <= 3  # max 3 players

model += sum(next_week_team) == 11  # total team size


# placeholder budget and prices
budget_now = 0
buy_prices = sell_prices = np.random.uniform(4, 12, size=100)

transfer_in_cost = sum(transfer_in_decisions[i] * buy_prices[i] for i in range(num_players))
transfer_out_cost = sum(transfer_in_decisions[i] * sell_prices[i] for i in range(num_players))

budget_next_week = budget_now + transfer_out_cost - transfer_in_cost
model += budget_next_week >= 0


# objective function:
model += sum((next_week_team[i]) * expected_scores[i]
             for i in range(num_players)), "Objective"



model.solve()
names = df["first_name"] + " " + df["second_name"]



num_players = 100
current_squad_indices = np.random.randint(0, num_players, size=15)
clubs = np.random.randint(0, 20, size=100)
positions = np.random.randint(1, 5, size=100)
expected_scores = np.random.uniform(0, 10, size=100)
current_squad_decisions = np.zeros(num_players) 
current_squad_decisions[current_team_indices] = 1
# placeholder budget and prices
budget_now = 0
buy_prices = sell_prices = np.random.uniform(4, 12, size=100)

opt = TransferOptimiser(expected_scores, buy_prices, sell_prices, positions, clubs)




transfer_in_decisions, transfer_out_decisions, starters, sub_decisions, captain_decisions = opt.solve(current_squad_indices, budget_now, sub_factor=0.2)


for i in range(num_players):
    if transfer_in_decisions[i].value() == 1:
        print("Transferred in: {} {} {}".format(i, buy_prices[i], expected_scores[i]))
    if transfer_out_decisions[i].value() == 1:
        print("Transferred out: {} {} {}".format(i, sell_prices[i], expected_scores[i]))



expected_scores = df["total_points"] / 38  # penalises players who played fewer games
prices = df["now_cost"] / 10
positions = df["element_type"]
clubs = df["team_code"]
# so we can read the results

decisions, captain_decisions, sub_decisions = select_team(expected_scores, prices.values, positions.values, clubs.values)
player_indices = []

print()
print("First Team:")
for i in range(len(decisions)):
    if decisions[i].value() == 1:
        print("{}{}".format(names[i], "*" if captain_decisions[i].value() == 1 else ""), expected_scores[i], prices[i])
        player_indices.append(i)
print()
print("Subs:")
for i in range(len(sub_decisions)):
    if sub_decisions[i].value() == 1:
        print(names[i], expected_scores[i], prices[i])
        player_indices.append(i)



# next week score forecast: start with points-per-game
score_forecast = df["total_points"] / 38
# let's make up a nonsense forecast to add some dynamics -- +1 to Chelsea players
score_forecast.loc[df["team_code"] == 8] += 1
# -1 for Liverpool players
score_forecast.loc[df["team_code"] == 14] -= 1
score_forecast = score_forecast.fillna(0)





opt = TransferOptimiser(score_forecast.values, prices.values, prices.values, positions.values, clubs.values)
transfer_in_decisions, transfer_out_decisions, starters, sub_decisions, captain_decisions = opt.solve(player_indices, budget_now=0, sub_factor=0.2)



if st.button('CLICK TO CHECK IF ANY TRANSFERS ARE NEEDED'):
    for i in range(len(transfer_in_decisions)):
        if transfer_in_decisions[i].value() == 1:
            st.write("TRANSFER IN: {} {} {}".format(names[i], prices[i], score_forecast[i]))
        if transfer_out_decisions[i].value() == 1:
            st.write("TRANSFER OUT: {} {} {}".format(names[i], prices[i], score_forecast[i]))



agree = st.checkbox('DO YOU WANT TO DO THE ABOVE TRANSFERS')

if agree:
    player_indices = []
    print()
    st.markdown("**PLAYING XI:**")
    st.write("GOALKEEPER")
    for i in range(len(starters)):
        if starters[i].value() != 0:
            
            if positions[i] == 1:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)

    st.write("DEFENDERS")
    for i in range(len(starters)):
        if starters[i].value() != 0:
            
            if positions[i] == 2:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)

    st.write("MIDFIELDERS")
    for i in range(len(starters)):
        if starters[i].value() != 0:

            if positions[i] == 3:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))

                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)

    st.write("FORWARDS")
    for i in range(len(starters)):
        if starters[i].value() != 0:

            if positions[i] == 4:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)
                
    print()
    st.markdown("**SUBSTITUTES:**")
    st.write("GOALKEEPER")
    for i in range(len(starters)):
        if sub_decisions[i].value() == 1:
            
            if positions[i] == 1:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)

    st.write("DEFENDERS")
    for i in range(len(starters)):
        if sub_decisions[i].value() == 1:
            
            if positions[i] == 2:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)

    st.write("MIDFIELDERS")
    for i in range(len(starters)):
        if sub_decisions[i].value() == 1:

            if positions[i] == 3:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))

                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)

    st.write("FORWARDS")
    for i in range(len(starters)):
        if sub_decisions[i].value() == 1:
            if positions[i] == 4:
                st.markdown("![Alt Text](https://resources.premierleague.com/premierleague/photos/players/110x140/p{}.png)".format(pic[i]))
                st.write("{} | Price = {}{}".format(names[i], prices[i], " [Captain] " if captain_decisions[i].value() == 1 else ""))
                player_indices.append(i)



st.markdown(
    """<a href="https://www.google.com/">example.com</a>""", unsafe_allow_html=True,
)
st.markdown('''
    <a href="https://www.google.com">
        <img src="https://media.tenor.com/images/ac3316998c5a2958f0ee8dfe577d5281/tenor.gif" />
    </a>''',
    unsafe_allow_html=True
)
   



