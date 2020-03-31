import streamlit as st
import plotly.express as px
import json
from scipy.integrate import odeint
from scipy.optimize import linprog
import plotly.figure_factory as ff
import re
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Constants 
REPRODUCTIVE_RATE = 2.28

countries = ['US','India','United Kingdom','Italy', 'China', 'Iran', 'Korea, South','Spain', 'Germany','Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Congo (Brazzaville)', 'Congo (Kinshasa)', 'Costa Rica', "Cote d'Ivoire", 'Croatia', 'Diamond Princess', 'Cuba', 'Cyprus', 'Czechia', 'Denmark', 'Djibouti', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Holy See', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Korea, South', 'Kuwait', 'Kyrgyzstan', 'Latvia', 'Lebanon', 'Liberia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malaysia', 'Maldives', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'San Marino', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Singapore', 'Slovakia', 'Slovenia', 'Somalia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Taiwan*', 'Tanzania', 'Thailand', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'Uruguay', 'US', 'Uzbekistan', 'Venezuela', 'Vietnam', 'Zambia', 'Zimbabwe', 'Dominica', 'Grenada', 'Mozambique', 'Syria', 'Timor-Leste', 'Belize', 'Laos', 'Libya']

def one_diff(val_1, val_2):
    try:
        x = (val_1 - val_2) / val_2
    except:
        x = 0
    return x


def seven_day_average(values):
    """
    Return a list of seven day averages
    """
    r_values = values[-7:]
    rate = [one_diff(r_values[1], r_values[0]), one_diff(r_values[2], r_values[1]), one_diff(r_values[3], r_values[2]), one_diff(r_values[4], r_values[3]), one_diff(r_values[5], r_values[4]), one_diff(r_values[6], r_values[5])]
    rate = [r*100 for r in rate]
    return rate


def rate(values):
    """
    "Taking only the last five values"
    
    :param values: [description]
    :type values: List of integers
    """
    r_values = values[-5:]
    rate = ((1/4) * (one_diff(r_values[1], r_values[0]) +  one_diff(r_values[2], r_values[1]) +  one_diff(r_values[3], r_values[2]) + one_diff(r_values[4], r_values[3]) )) * 100
    return rate

@st.cache
def core_data():
    return requests.get('https://pomber.github.io/covid19/timeseries.json')
    
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def find_closest_number(num,collection):
   return min(collection,key=lambda x:abs(x-num))

def get_corona_data(country_name):
    r = core_data()
    x = r.json()
    # with open('timeseries.json') as fp:
    #     x = json.load(fp)
    country_stats = x[country_name]
    dates = [ datetime.strptime(d['date'], "%Y-%m-%d") for d in country_stats]
    deaths = [d['deaths'] for d in country_stats]
    confirmed = [d['confirmed'] for d in country_stats]
    recovery = [d['recovered'] for d in country_stats]
    idx = next((i for i, x in enumerate(confirmed) if x), None)
    date_of_spread_start = dates[idx]
    case_rate = rate(confirmed)
    death_rate = rate(deaths)
    recovery_rate = rate(recovery)
    seven_day_cases_pc, seven_day_deaths_pc = seven_day_average(deaths), seven_day_average(confirmed)
    return dates, deaths, confirmed, recovery, case_rate, death_rate, recovery_rate, date_of_spread_start, seven_day_cases_pc, seven_day_deaths_pc

def sir_model(total_population,infected,recovered, contact_rate, recovery_rate, death_rate):
    S0 = total_population - infected - recovered
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    beta, gamma = contact_rate, recovery_rate - death_rate
    # A grid of time points (in days)
    t = np.linspace(0, 365, 365)


with open('country_population.json') as fp:
    pop_data = json.load(fp)


st.title("When will Coronavirus COVID-19 crisis be over?")
st.markdown("")
st.markdown("We are scared, concerned, and locked down. The question every one is asking is - _when will this be over_. There's also a `worry index` in this post, which should help with our panic buying impulses.")


st.sidebar.header("This calculator will help us get to a reasonable estimate, step-by-step. ")


st.sidebar.markdown("Please answer these questions. Feel free to explore other countries, or different scenarios. Note that the numbers and text in the adjacent blog post change based on your responses here.")

user_country = st.sidebar.selectbox("Your Country", countries, 0)
population = pop_data[user_country]['population']
country_rank = pop_data[user_country]['rank']
population_density = pop_data[user_country]['pop_density']
st.sidebar.markdown("##### The community outside")
lockdown_status = st.sidebar.radio("How strictly is lockdown/shelter-in-place being enforced?", ["no one's allowed out", "you see only a few people out", "people are secretly not following rules", "people are openly not following rules"])

# st.sidebar.markdown("##### The Human Network and Health System Response")
# st.sidebar.markdown("> Play with this section to see how governmental initiatives, such as contact tracing, can affect the rate of infected and thus, the death rate")
# network_degree = st.sidebar.slider(f"On average, how many less number of people an individual is meeting due to lockdowns and social distancing? (For example, if one used to meet around 4 people at work on average every day, so for her - the value would be 4. What's the typical in {user_country}) ", min_value=1, max_value=20, value=3, step=1)

network_degree = 4
# worst_contact_trace = st.sidebar.selectbox("Worst Case Contact Tracing", ["No one", "Immediate caregiver only", "All family members"], 0)

# if worst_contact_trace == "No one":
#     worst_contact_trace = 0
# elif worst_contact_trace == "Immediate caregiver only":
#     worst_contact_trace = 1
# elif worst_contact_trace == "All family members":
#     worst_contact_trace = 2
# else:
#     worst_contact_trace = 0

worst_contact_trace = 1


# st.sidebar.markdown("##### My Impact on flattening the curve")
# st.sidebar.markdown("> Play with this question and the slider to see how YOU impact your country")
# personal_lockdown_belief = st.sidebar.slider("How much social distancing, isolation, lockdown, or shelter-in-place are you practicing? (0 if none at all, 100 if you are following all best practices)", value=80)
# personal_lockdown_belief = (100 - personal_lockdown_belief) / 110

st.sidebar.markdown("##### Do I need help?")
current_health = st.sidebar.radio("Are you suffering from any of the following now?", ["None","Runny Nose, Minor Coughs","Lost Smell+Fever+Cough Fits", "Runny Nose+Fever+Cough Fits+Loose Motion"])
travel_history = st.sidebar.checkbox("Check this if you have traveled to any of the known outbreak regions in the last three weeks?")

st.sidebar.markdown("##### People around me")
hygiene_status = st.sidebar.radio("Are most people you know washing their hands often?", ["No", "Yes"])
age_group = st.sidebar.slider("Age group of your close family & friends (years), including yourself", 0, 120, (9, 45))
preexisting_conditions = st.sidebar.slider("Number of people (including yourself) you know who have prexisting conditions such as diabetes, cancer, high blood pressure, and other immuno-deficiency diseases")
min_age = min(age_group)
max_age = max(age_group)

if current_health == "None":
    current_health_index = 0
elif current_health == "Runny Nose":
    current_health_index = 2
elif current_health == "Lost Smell+Fever+Coughing Fits":
    current_health_index = 4
elif current_health == "Runny Nose+Fever+Cough Fits+Loose Motion":
    current_health_index = 8
else:
    current_health_index = 1

if travel_history:
    travel_history_index = 4
else:
    travel_history_index = 1

if any([ min_age < 10, 50 <=max_age < 60 ]):
    age_group_risk_weight = 2
elif any([ min_age < 10, 60 <=max_age < 65 ]):
    age_group_risk_weight = 3
elif any([min_age < 10,  max_age >= 65]):
    age_group_risk_weight = 4
else:
    age_group_risk_weight = 1

if preexisting_conditions > 0:
    preexisting_conditions_weight = 2
else:
    preexisting_conditions_weight = 1


if lockdown_status == "no one's allowed out":
    lockdown_intensity_value = 0.05
elif lockdown_status == "you see only a few people out":
    lockdown_intensity_value = 0.10
elif lockdown_status == "people are secretly not following rules":
    lockdown_intensity_value = 0.33
elif lockdown_status == "people are openly not following rules":
    lockdown_intensity_value = 0.66

dates, deaths, confirmed, recovery, case_rate, death_rate, recovery_rate, date_of_spread_start, seven_day_cases_pc, seven_day_deaths_pc = get_corona_data(user_country)

try:
    death_to_case_ratio = deaths[-1] / confirmed[-1]
except:
    death_to_case_ratio = -1

lockdown_impact = lockdown_intensity_value #(1/11)*(10*lockdown_intensity_value + personal_lockdown_belief)
beta = (REPRODUCTIVE_RATE*lockdown_impact + case_rate/100)/3.28
gamma = 1/14
t = list(np.linspace(0, 730, 730))
# Initial conditions vector
population =  int(population.replace(",",""))
I0 = confirmed[-1]
R0 = recovery[-2] + deaths[-2]
S0 = population - I0 - R0
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(population, beta, gamma))
S, I, R = ret.T

total_susceptibles = max(S)
total_infections = max(I)
total_recovered = sum(R)

total_deaths = death_to_case_ratio * total_infections
total_survivals = population - (total_infections)
total_no_corona = population - total_susceptibles - total_recovered

I = list(I)
S = list(S)
I = [int(i) for i in I]
I_max = max(I)

I_5 = I[4]
probability_of_getting_infected = I_5 / population
probability_of_getting_infected_max = I_max / population
I_max_index = I.index(I_max)
post_peak_I = I[I_max_index:]
safe_number = find_closest_number(int(I_max * 0.377), post_peak_I)
safe_number_index = post_peak_I.index(int(safe_number))
days_left = I_max_index + safe_number_index
# days_left = t[days_left]

today = datetime.today()
good_news_date = today + timedelta(days=days_left)
good_news_date = good_news_date.strftime("%B %d, %Y")


susceptible = S0
infected = I0


def optimization_model(country_rank, population, susceptible, infected, lockdown_intensity_value,total_susceptibles,total_infections, network_degree,worst_contact_trace):
    population = int(population)
    if country_rank == "not the best":
        health_care_capacity_percent = 0.02/100
        max_sick_people = health_care_capacity_percent * population
    else:
        country_rank = int(country_rank)
        if country_rank <= 30:
            health_care_capacity_percent = 0.2/100
            max_sick_people = health_care_capacity_percent * population
        elif 31 <= country_rank <= 50:
            health_care_capacity_percent = 0.15/100
            max_sick_people = health_care_capacity_percent * population
        elif 51 <= country_rank <= 75:
            health_care_capacity_percent = 0.12/100
            max_sick_people = health_care_capacity_percent * population
        elif 75 <= country_rank <= 100:
            health_care_capacity_percent = 0.10/100
            max_sick_people = health_care_capacity_percent * population
        elif country_rank >= 100:
            health_care_capacity_percent = 0.08/100
            max_sick_people = health_care_capacity_percent * population
        else:
            health_care_capacity_percent = 0.05/100
            max_sick_people = health_care_capacity_percent * population

    # print(total_susceptibles, total_infections)
 
    c = [lockdown_intensity_value*REPRODUCTIVE_RATE, -1*network_degree*REPRODUCTIVE_RATE]
    A = [[-1*network_degree*REPRODUCTIVE_RATE, 2*REPRODUCTIVE_RATE],[ (0.1)*REPRODUCTIVE_RATE, -1*network_degree*REPRODUCTIVE_RATE]]
    b = [total_susceptibles, max_sick_people]
    x0_bounds = (0, total_susceptibles)
    x1_bounds = (infected, total_infections)
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
        res = res['x']
        # print(res)
    except:
        res = False
    return res, max_sick_people


res, max_sick_people = optimization_model(country_rank, population, susceptible, infected, lockdown_intensity_value, total_susceptibles, total_infections, network_degree,worst_contact_trace)

if res is not False:
    actual_susceptible = int(res[0])
    actual_infections = int(res[1])
else:
    print("Here")
    actual_susceptible = int(total_susceptibles)
    actual_infections = int(total_infections)




peak_date = datetime.today() + timedelta(days=I_max_index)
peak_date = peak_date.strftime("%B %d, %Y")
S_peak = max(S)
S_safe = 0




worry_index = lockdown_intensity_value * (1/6) * ( 2 * int(population_density.replace(',','')) /750 + 3 * case_rate /100 + (death_rate - recovery_rate)/100)




if death_to_case_ratio <= 0.01:
    death_to_case_ratio_qual = "is low, so don't worry too much"
elif 0.01 <= death_to_case_ratio < 0.03:
    death_to_case_ratio_qual = "is not very high"
elif 0.03 <= death_to_case_ratio < 0.07:
    death_to_case_ratio_qual = "is on the higher side"
else:
    death_to_case_ratio_qual = "is very high"

# st.markdown(f"Today is day {len(dates)} of COVID-19.")




chart_data = pd.DataFrame({
    'Deaths': deaths,
    'Confirmed': confirmed}
)

st.line_chart(chart_data)






if case_rate > 1.25:
    x = int(days_left / 7)
    if x > 100:
        st.markdown(f"> Our model suggests it will take __almost two years__, about {good_news_date}, for everything to be safe and normal again. In this period, there will be approximately {actual_infections:,} cases in {user_country}. Most of these cases will be minor. However, the number of __lives lost__ _can_ be as high as __{int(actual_infections*death_to_case_ratio):,}__.")
    else:
        st.markdown(f"> Our model suggests that, __in {x} weeks, that is, by {good_news_date}__, everything will be safe and normal again. There is a risk of {actual_infections:,} cases in {user_country}. Most of these cases will be minor. However, the number of __lives lost__ _can_ be as high as __{int(actual_infections*death_to_case_ratio):,}__.")
else:
    st.markdown(f"> Our model suggests that, the pandemic is not a major crisis in {user_country} at this moment. Still there is a risk of {actual_infections:,} cases in {user_country}. Most of these cases will be minor. It is still important to note that, the number of __lives lost__ _can_ be as high as __{int(actual_infections*death_to_case_ratio):,}__, as well as chance of a new outbreak.")

st.markdown(f"Of these {actual_infections:,} cases, about {int(0.2*actual_infections):,} will require some degree of hospitilization. The health care system in {user_country} is prepared for approximately {int(max_sick_people):,} people.")

st.markdown("## What was the daily percent change in cases and deaths, over the last week?")

days = [today - timedelta(days=5), today - timedelta(days=4), today - timedelta(days=3), today - timedelta(days=2), today - timedelta(days=1), today]
days = [d.strftime("%b %d") for d in days]
chart_data = pd.DataFrame({
    "Days": days,
    'Cases %': seven_day_cases_pc,
    'Deaths %': seven_day_deaths_pc}
)
chart_data = chart_data.rename(columns={'Days':'index'}).set_index('index')
st.line_chart(chart_data)


st.markdown("## Social / physical distancing changes the blue curve below, which changes everything")

st.markdown(f"Feel free to adjust the level of social/physical distancing being practiced in {user_country} to see how the curve and the numbers change.")

chart_data = pd.DataFrame({
    'People at Risk': S,
    'Confirmed Cases': I,
    'Removed from Risk': R}
)

st.line_chart(chart_data)

st.markdown(f"- Estimated number of cases: __{actual_infections:,}__")
st.markdown(f"- Estimated number of lives lost: __{int(actual_infections*death_to_case_ratio):,}__")


st.write("""
 This model:
- assumes no medicine/vaccine/intervention for COVID-19 is found
- assumes weather has no effect on this novel coronavirus
- assumes countries are homogeneous, which they are not. Real risks and numbers will depend upon cities, geographies, and local policies.
- may show astronomical, scary numbers - which are pessimitically plausible if medical interventions like preventative drugs or vaccines do not become available soon, or warmer weather doesn't help
- is intended to be a curiosity tool, not a decision making one

Please feel free to play around with other parameters to see the impact of the virus.
""")


st.markdown(f"## What will be the number of cases on ... ?")

interest_date = st.date_input("Choose a date", datetime.today())

my_time = datetime.min.time()
interest_date = datetime.combine(interest_date, my_time)
diff = (interest_date - today).days
if diff < 0:
    if abs(diff) <= len(confirmed):
        write_string = f"As of {interest_date.strftime('%b %d, %Y')}, there were {int(confirmed[diff]):,} known cases."
    else:
        write_string = f"Covid-19 was not known as a threat as of {interest_date.strftime('%b %d, %Y')}."
else:
    if diff > len(I):
        diff = len(I)
    x = 0

    new_cases = int(I[diff])
    
    cases_on_interest_date = int(I[diff])
    expected_fatality = cases_on_interest_date * death_to_case_ratio

    write_string = f"As of {interest_date.strftime('%b %d, %Y')}, we expect {new_cases:,} total cases."

st.markdown(write_string)



st.markdown(f"## How much should you be concerned?")
st.markdown("Even though you are managing panic, some concern is unavoidable. Also, everybody need not worry equally. There are two main types of concern people are enduring in such times:")
st.markdown("- Concern for the country and economy")
st.markdown("- Concern for self, family and friends")

st.markdown(f"#### Concern for the country")

if int(recovery[-2]) != 0:
    st.markdown(f"First, the good news is, {recovery[-2]} have recovered since the outbreak in {user_country}.")

date_of_spread_start_string = date_of_spread_start.strftime("%b %d, %Y")
if int(death_to_case_ratio * 100) == 0:
    st.markdown(f"{user_country} has about {population} people. Approximately, {population_density} people live per sq mile. So far, there are {confirmed[-1]} cases of COVID-19 in {user_country}, since the virus began spreading here on {date_of_spread_start_string}. The rank of health infrastructure in {user_country} is  {country_rank} in the world, according to [Wikipedia](https://en.wikipedia.org/wiki/World_Health_Organization_ranking_of_health_systems_in_2000). Overall, the mortality rate is {death_to_case_ratio_qual}.")
else:
    st.markdown(f"{user_country} has about {int(int(population)/1000000):} million people. Approximately, {population_density} people live per sq mile. So far, there are {confirmed[-1]} cases of COVID-19 in {user_country}, since the virus began spreading here on {date_of_spread_start_string}. The mortality rate {death_to_case_ratio_qual}. The health infrastructure in your country is ranked {country_rank} in the world, according to [Wikipedia](https://en.wikipedia.org/wiki/World_Health_Organization_ranking_of_health_systems_in_2000). Numerically, {int(death_to_case_ratio*100)} in 100 affected people have been killed by the virus.")
st.markdown(f"In the _last five days_, the death rate has been {death_rate:.2f} %, while the number of confirmed cases have been increasing by {case_rate:.2f}% day on day.")


st.markdown(f"- As a whole nation, people of {user_country} can be attributed a__ worry index score of {worry_index:.3f}__, where values close to 0 are good, and _if_ it is close to 1 - you should take more precautions.")
st.markdown(f"- In the _next five days_, your chance of getting an infection is __{probability_of_getting_infected*100:.2f}%__. This is because  {lockdown_status}.")
st.markdown(f"- In the total period of crisis, the peak probability of getting infected will come in about {int(I_max_index/7)} weeks. During the __week of {peak_date}__, the chance that a previously un-affected person will get COVID-19 is expected to be  __{probability_of_getting_infected_max*100:.2f}%__. ")
st.write("""
This is how we calculated the worry index: 
$$
WI_{national} = w_L \\times \left( \\frac{w_\\rho \\rho_{normal} + w_{b} b + w_{\\Gamma} \\Gamma }{\sum w} \\right)
$$
where $w_L$ is a weight associated with the severity of the lockdown intensity, $\\rho_{normal}$ is the population density normalized to 500 people per square mile, $w_\\rho$ is the weight of the population density, $b$ is the rate of increase of confirmed cases, $w_b$ is the weight associated with rise in confirmed cases,$\gamma$ is the recovery and death rate,  $w_\\Gamma$ is the weight associated with death and recovery rates, $w$ is the set of all weights.
""")

st.markdown(f"#### Concern for self, family, and friends")


if hygiene_status == "Yes":
    personal_worry_index = 0.3 * (age_group_risk_weight / 4 + preexisting_conditions_weight / 2)
else:
    personal_worry_index = 0.75 * (age_group_risk_weight / 4 + preexisting_conditions_weight / 2)


personal_worry_index_total = 1 / 10 * (worry_index + 4*personal_worry_index + 2*(1/8)*current_health_index + 4*(1/4)*travel_history_index)
if personal_worry_index_total < 0.33:
    personal_worry_index_suggestion = "You have little to worry about, statistically speaking. However, you must continue to exert utmost care and precautions, such as handwashing and social distancing."
    worry_message_special = "You have almost nothing to worry about. Don't hoard a lot of food and essential supplies, more disadvantaged people might/will need them in the coming days."
elif 0.33 <= personal_worry_index < 0.5:
    personal_worry_index_suggestion = "You'll be fine if you  excercise a lot more caution than the average individuals around you. Alongside social distancing, try to stay as much indoors as possible, and clean surfaces that you touch frequently with disinfectants. If you had to go out for some compelling reason, change your clothes immediately upon return."
    worry_message_special = "Exercise abundant caution, and if possible, get hold of a few masks and gloves. "
elif 0.5 <= personal_worry_index < 0.66:
    personal_worry_index_suggestion = "You must take extra precaution. You should not venture out of your home. Keep contact numbers of family and medical professionals handy. Also, share your situation and concerns with close family and friends, as best as you can. Please maintain strict social distancing and stick to best practices recommended by health officials."
    worry_message_special = "Please take extra care, as things might escalate quickly. Feel free to stock up on some food and protective gear items if you can."
else:
    personal_worry_index_suggestion = "If you are experiencing any physical or mental difficulty, you should consider seeking advice of family and friends. Make sure you maintain strict social distancing under all conditions, unless the other person is wearing protective gear."
    worry_message_special = "Please try and get in touch with someone for physical or emotional help."
st.markdown(f"On the personal front, we project the worry index for you to be __{personal_worry_index_total:.3f}__. {personal_worry_index_suggestion}")

st.markdown(f"## What's the point of all these lockdowns and social distancing?")

# st.markdown("Thank you for doing your part and doing you part in helping the country and our civilization. Lockdowns and social distancing are non-medical interventions to manipulate the reproductive rate of the novel coronavirus.")
st.markdown("If left unchecked - i.e. the normal life scenario - this menacing virus will affect nearly 3 people for every one infected person (as of now, the estimated reproductive rate of the virus is 2.28).")
st.markdown("If we don't do these lockdowns, we will get over this crisis faster, BUT, at the disastrous cost of lives. But, if we do the lockdowns, the lives lost will be much less - but, the period will stretch longer. Check out the numbers by choosing the different social distancing enforcement options")




st.markdown("## How did we calculate the crisis end-date?")



st.markdown("We are using a modified version of SIR model, proposed by Kermack and McKendrick in 1927. The _SIR_ model stands for spread or susceptibility (S), infection rate (I), and removal (death) or recovery rate (R). This model was proposed during the period following the similar Spanish Flu crisis of 1918.")
st.write("""
According to a pre-print by S.Zhao, published in March 2020, the reproductive number ($R_0$) of the novel coronavirus is estimated to be 2.28. It means, each person infected with the virus can pass it onto 2.28 individuals. 
Thus, the effective contact rate of the disease can be modeled as:

$$
\\beta = \\textrm{Rate of Confirmed Cases} \\times R_0 \\times \\textrm{Impact of Lockdowns}
$$

Thus, the change in number of confirmed cases (i.e. suspects, S) with respect to time can be modeled by the following differential equation:

$$
\\frac{dS}{dt} = - \\frac{\\beta SI}{\\textrm{Total population size (N)}}
$$
Similarly, the models for the infection and recovery rates ($\gamma$)are: 

$$
\\frac{dI}{dt} = \\frac{\\beta SI}{N} - \gamma I 
$$

Also, we are assuming that the recovery time of COVID-19 is about 14 days for most survivors, during which they can infect others. Thus, the $1/\gamma$ is 14. 


We hypothesize that it will be safe to return to normalcy, when the number of confirmed cases per day has fallen to 37.7 percent of the peak value. 
""")


st.markdown("### A simple optimization model")
st.write("""

The officials will try to follow an optimization model, though sticking to it will not be always possible. The goal is to minimize the number of susceptible ($S$) and infected ($I$) people at the same time. Our primary constraint is our health care system - the doctors, the nurses, the paramedics, the ICUs, and such. The real problem formulation is more complex, but here,we are trying to model the bare minimum version possible. 

There are two factors dominating here:

- The number of people each susceptible person (say, $x_0$) can infect is proportional to the reproductive rate, and the amount of social distancing they practice ($s$)
- The number of people infected (say, $x_1$), and their family members are almost immediately quarantined and isolated (quarantine factor, $q_1$)- stopping the spread of the virus. 

Together, these two parameters constitute the pressure facing the health care system, which must be minimized. The cost function becomes:

$$
2.28sx_0 - 2.28q1*x_1 
$$

A comprehensive health care capacity (say $H_{max}$) across the world is not available, and is changing dynamically. So, we are using an approximation based on the overall health care rank of the country:
- between 1 to 30 - we _assume_ the country has the capability to handle 0.2 percent of the population based on ball park figures from a sample of countries within these rank ranges and their respective populations:
- between 31 to 50 - 0.15 percent
- between 50 to 75 - 0.12 percent
- between 75 to 100 - 0.1 percent
- beyond 100 - 0.08 percent or lower


Assuming the worst case scenario, where people are least respectful of social distancing, hygeine and quarantines:
$$
2.28s_{min}x_0 - 2.28*q^1_{min}x_1 > S
$$
And, the best possible scenario, where people are appropriately quarantined and follow the maximum possible social distancing measures:
$$
2.28s_{max}x_0 - 2.28*q^1_{max}x_1 < H_{max}
$$

The bounds on the variables are put as: 
$$
\\textrm{People at risk today} < x_0 < \\textrm{Max susceptible people}
$$

$$
\\textrm{People already sick} < x_1 < \\textrm{Max infectible people}
$$

By solving this optimization problem, we estimate the optimal number of sick and infected people we can expect in our society during the peak periods of crisis.

""")




st.markdown("### Finally, remember ...")
st.markdown(f"- In {user_country}, expect normalcy to gradually return by {good_news_date}")
st.markdown(f"- The worst part of the crisis will be in the week of {peak_date}")
st.markdown(f"- In the next five days, you have a {probability_of_getting_infected*100:.3f}% chance of contracting coronavirus, that can lead to COVID-19 complications.")
st.markdown(f"- Personally, your worry index for self, family, and friends is {personal_worry_index_total:.3f}. {worry_message_special}")

# st.markdown("This is a dynamic, interactive blog post - which means numbers change with the situation in your country. Please check back often for the latest.")
# st.markdown(f"- Please discourage people who propagate conspiracy theories, and posts which spread divisive or racist myths against any country or any ethnicity. This is a human, global crisis. We can only solve this if we stand against it together, shoulder to shoulder ... well, while maintaining a 3 feet distance, for now!")
# st.markdown(f"- If you have something to spare, please donate generously to [WHO Covid 19 Response Fund](https://covid19responsefund.org/). There are many more people in much worse conditions than who can read this. Your donations are tax-deductible.")
# st.markdown("- Most importantly, stay safe. Please follow _only_ WHO, CDC, and local health official websites and advisories only for advice.")
# st.markdown("")
st.markdown("### End Notes")
st.markdown("- _This is a very simplified model, for general outlook only_. It is only slightly better than nothing but guesswork.")
st.markdown("- This model uses the latest WHO data, accessed via an open source API")
st.markdown("- My goal is to help people make slightly-more informed decisions, erring on the side of caution. I am very open to making a lot of changes in the model as necessary. Also, if you have any questions or suggestions on improving this model, please contact me via email: [chanda.sayonsom@gmail.com](mailto:chanda.sayonsom@gmail.com), or using a LinkedIn message on my profile: [Sayonsom Chanda](https://www.linkedin.com/in/sayonsom) ")
st.markdown("- This is an open-source project hosted on [Github](https://www.github.com/sayonsom/covid19_when_will_it_be_over)")
# , on a Creative Commons License. Please feel free to improve this model, translate this interactive blog post into another language, [fork it, submit a PR, or leave a star](https://www.github.com/sayonsom/covid19_when_will_it_be_over). Also [email me](mailto:chanda.sayonsom@gmail.com) if you need more details.")

