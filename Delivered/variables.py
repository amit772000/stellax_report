import pandas as pd
address="Kastelenstraat 213-1, 1082 EE Amsterdam"
address_short="Kastelenstraat 213-1"
city="Amsterdam"
report_date="19.06.2025"
reference_date="01.01.2025"
report_link="https://drive.google.com/drive/folders/1W16DAJx3sVnM26RHR-kjDJ0ouoLJ6SFm"
vacant_value=1100000 #can be anywhere between 50000 and 10000000
rented_value=1200000 #can be anywhere between 50000 and 10000000
market_rent=900 #can be anywhere between 300 and 10000
wws_points=80 #can be anywhere between 30 and 500
wws_points_rent=900 #can be anywhere between 200 and 5000

property_type="Apartment"
sqm=130 #can be anywhere between 10 and 800
year=1963
lot_size=0 #can be anywhere between 0 and 10000
energy_label="-"
contract_rent=300 #can be anywhere between 300 and 10000
vve=90 #can be anywhere between 0 and 2000
erfpact_date="-"
erfpacht_amount="1000"

property_type_source="Kadaster (BAG)"
sqm_source="Kadaster (BAG)"
year_source="Kadaster (BAG)"
lot_size_source="Kadaster (BAG)"
energy_label_source="Kadaster (BAG)"
contract_rent_source="Kadaster (BAG)"
wws_points_source="Kadaster (BAG)"
wws_rent_source="Kadaster (BAG)"
vve_source="Kadaster (BAG)"
erfpact_date_source="Kadaster (BAG)"
erfpacht_amount_source="Kadaster (BAG)"

property_overview_1="This apartment of 79 m² is located in Buitenveldert, a popular area known for its residential character and livability. With an energy label of C, the property offers reasonable energy performance, contributing to lower energy costs and increased appeal."
property_overview_2="The property has been evaluated across multiple dimensions including its vacant market value, value in rented state, rental potential, and WWS score, all benchmarked against local market data and similar homes in the area."
#property_google_photo.png #<-------------------------------------
#cadastral_map.png #<-------------------------------------

vacant_value_score=0.07 #can be anywhere between 0% and 100%
vacant_value_low=990000 #can be anywhere between 50000 and 10000000
vacant_value_high=1200000 #can be anywhere between 50000 and 10000000

data = {
    "Sqm": [
        45, 50, 55, 58, 60, 62, 65, 67, 70, 72,
        75, 78, 80, 82, 85, 88, 90, 92, 95, 98,
        100, 105, 108, 110, 115, 118, 120, 125, 130, 135, 140, 200
    ],
    "Price_m2": [
        4000, 5000, 7700, 7600, 7450, 7200, 7100, 6900, 6700, 6600,
        6400, 6300, 6150, 6000, 4900, 5750, 5650, 5600, 5500, 5400,
        5300, 5200, 6300, 5100, 5050, 5000, 4950, 4900, 4850, 4800, 4700, 10000
    ]
}

vacant_values_per_sqm_comps_df = pd.DataFrame(data)



data = {
    "Address": [
        "Koxhorn 10 2, 1082EV Amsterdam",
        "Koxhorn 11 3, 1082EV Amsterdam",
        "Zuid-Hollandstraat 60 2, 1082EL Amsterdam",
        "Boeckenburg 3, 1082CT Amsterdam",
        "Koxhorn 23 2, 1082EV Amsterdam"
    ],
    "Distance_meters": [200, 9, 9, 9, 9],
    "Type": ["Apartment"] * 5,
    "Asking_price": [1000000, 1000000, 1000000, 1000000, 1000000],
    "Bid_above_asking_pct": [5.4, 11.8, 12.2, 0.0, 1.7],
    "Adjusted_price": [1000000, 1000000, 1000000, 1000000, 1000000],
    "Square_meters": [80, 77, 78, 85, 79],
    "Lot_size": [0, 0, 0, 0, 0],
    "Year": [1962, 1962, 1963, 1963, 1962],
    "Date": ["Q3 2024", "Q2 2024", "Q2 2024", "Q2 2024", "Q4 2023"]
}

vacant_values_comps_df = pd.DataFrame(data)

#vacant_values_comps_df.png #<-------------------------------------

vacant_value_optimal_asking=1100000 #can be anywhere between 50000 and 10000000
vacant_value_final_price_paid=1200000 #can be anywhere between 50000 and 10000000


data = {
    "Price_paid": [1100000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000],
    "Asking_price": [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000]
}

asking_vs_price_paid_df = pd.DataFrame(data)


vacant_value_optimal_asking_low=1200000 #can be anywhere between 50000 and 10000000
vacant_value_optimal_asking_high=1500000 #can be anywhere between 50000 and 10000000

percent_sold_above_asking=0.09 #can be anywhere between 0 and 0.99
average_bidding=0.12 #can be anywhere between 0 and 0.99


data = {
    "Bid_offered": [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000],
    "Chance_of_winning_pct": [5, 13, 28, 47, 66, 81, 91, 96]
}

bid_vs_winning_chance_df = pd.DataFrame(data)


vacant_value_demand_score=0.09 #can be anywhere between 0 and 1
vacant_value_market_descitption="The market is cold!"


data = {
    "Date": [
        "May-24", "Jun-24", "Jul-24", "Aug-24", "Sep-24", "Oct-24",
        "Nov-24", "Dec-24", "Jan-25", "Feb-25", "Mar-25", "Apr-25", "May-25"
    ],
    "Price_index": [
        100.0, 100.6, 101.1, 101.6, 102.1, 102.7,
        103.1, 103.2, 103.2, 103.2, 103.5, 104.3, 105.1
    ]
}

vacant_value_index_df = pd.DataFrame(data)
vacant_value_index_reference_date="05/24=100"


market_rent_score=0.89 #can be anywhere between 0% and 100%
market_rent_low=300 #can be anywhere between 300 and 10000
market_rent_high=2934 #can be anywhere between 300 and 10000


data = {
    "Sqm": [
        45, 50, 55, 60, 62, 65, 67, 70, 72, 75,
        78, 80, 82, 85, 88, 90, 92, 95, 98, 100,
        105, 110, 115, 118, 120, 125, 128, 130, 135, 140, 145
    ],
    "Rent_m2": [
        20, 44, 42, 39, 41, 38, 35, 37, 16, 36,
        100, 29, 11, 28, 26, 25, 22, 23, 32, 20,
        22, 24, 86, 21, 20, 22, 43, 25, 23, 26, 27
    ]
}

market_rent_per_sqm_comps_df = pd.DataFrame(data)


data = {
    "Address": [
        "Koxhorn 10 2, 1082EV Amsterdam",
        "Koxhorn 11 3, 1082EV Amsterdam",
        "Zuid-Hollandstraat 60 2, 1082EL Amsterdam",
        "Boeckenburg 3, 1082CT Amsterdam",
        "Koxhorn 23 2, 1082EV Amsterdam"
    ],
    "Distance_meters": [90, 87, 124, 133, 39],
    "Type": ["Apartment"] * 5,
    "Asking_price": [479000, 475000, 500000, 625000, 450000],
    "Bid_above_asking_pct": [5.4, 11.8, 12.2, 0.0, 1.7],
    "Adjusted_price": [512000, 568000, 596000, 613000, 511000],
    "Square_meters": [80, 77, 78, 85, 79],
    "Lot_size": [0, 0, 0, 0, 0],
    "Year": [1962, 1962, 1963, 1963, 1962],
    "Date": ["Q3 2024", "Q2 2024", "Q2 2024", "Q2 2024", "Q4 2023"]
}

market_rent_comps_df = pd.DataFrame(data)

#market_rent_comps_df.png #<-------------------------------------

market_rent_demand_score=0.09 #can be anywhere between 0 and 1
market_rent_market_descitption="The market is hot!"
#market_rent_market_emoji.png

data = {
    "Date": [
        "May-24", "Jun-24", "Jul-24", "Aug-24", "Sep-24", "Oct-24",
        "Nov-24", "Dec-24", "Jan-25", "Feb-25", "Mar-25", "Apr-25", "May-25"
    ],
    "Rent_index": [
        100.0, 100.4, 100.7, 100.9, 101.0, 101.1,
        101.2, 101.3, 101.5, 101.9, 102.6, 103.6, 105.3
    ]
}

market_rent_index_df = pd.DataFrame(data)
market_rent_index_reference_date=" 05/24=100"

wws_points_threshold=186
sector_text=" free "

wws_points_breakdown_dict={
    "Oppervlakte van vertrekken": 79,
    "Oppervlakte overige ruimten": 8,
    "Verwarming en installaties": 13,
    "Energieprestatie": 15,
    "Keuken": 10,
    "Sanitair": 10,
    "Voorzieningen gehandicapten": 0,
    "Privé-buitenruimten": 5,
    "WOZ-waarde": 57,
    "WOZ punten correctie": 0,
    "Renovatie": 0,
    "Zorgwoning": 0,
    "Gemeenschappelijke ruimten": 0,
    "Monument rent increase": "+10%"
}

gross_yield=0.11 #can be anywhere between 0.5% and 10.9%
net_yield=0.1 #can be anywhere between 0.5% and 10.9%
return_on_equity=0.108 #can be anywhere between 0.5% and 10.9%
cashflow=5000 #can be anywhere between -9000 and 30000

data = {
    "LTV": ["50%", "60%", "70%", "80%", "90%"],
    "4.5%": [3.8, 3.6, 3.4, 3.1, 2.4],
    "5.0%": [3.5, 3.2, 2.8, 2.2, 1.1],
    "5.5%": [3.2, 2.7, 2.2, 1.1, -0.1],
    "6.0%": [2.8, 2.3, 1.5, 0.4, -1.5],
    "6.5%": [2.5, 1.8, 0.8, -0.5, -2.9]
}

return_on_equity_df = pd.DataFrame(data)


data = {
    "LTV": ["50%", "60%", "70%", "80%", "90%"],
    "4.5%": [1112, 1054, 995, 907, 703],
    "5.0%": [1025, 937, 820, 644, 322],
    "5.5%": [937, 790, 644, 322, -29],
    "6.0%": [820, 673, 439, 117, -439],
    "6.5%": [732, 527, 234, -146, -849]
}

monthly_cash_flow_df = pd.DataFrame(data)

bar_kk=0.109 #can be anywhere between 0.5% and 10.9%
nar_kk=0.10 #can be anywhere between 0.5% and 10.9%

capitalisation_factor=20 #can be anywhere between 5 and 99
vacant_value_ratio=0.85 #can be anywhere between 20% and 100%
rent_vacant_value_ratio=0.1 #can be anywhere between 0.5% and 10.9%

effective_rent_yearly=27000
effective_rent_per_sqm=28.48
vacant_value_per_sqm=6722

market_rent_yearly=28716
contract_rent_yearly=27000
wws_rent_yearly=14928
effective_rent_method="Contracthuur"


municipality_taxes=504
management_costs=1080
maintenance_costs=808
VVE_yearly=1536
other_running_costs=0
total_running_costs=3992

running_costs_to_effective_rent_percentage=0.08
net_rental_income_to_effective_rent_percentage=0.92

net_rental_income=23072
nar_von=0.1
rented_value_von=505075

legal_and_delivery_costs=3859
transfer_tax=47216
other_costs_corrections=0


#energy_label_A+++++.png #<-------------------------------------
#energy_label_A++++.png #<-------------------------------------
#energy_label_A+++.png #<-------------------------------------
#energy_label_A++.png #<-------------------------------------
#energy_label_A+.png #<-------------------------------------
#energy_label_A.png #<-------------------------------------
#energy_label_B.png #<-------------------------------------
#energy_label_C.png #<-------------------------------------
#energy_label_D.png #<-------------------------------------
#energy_label_E.png #<-------------------------------------
#energy_label_F.png #<-------------------------------------
#energy_label_G.png #<-------------------------------------
#energy_label_NA.png #<-------------------------------------

energy_label_register_dat="27-07-2022"
energy_label_expiration_date="27-07-2032"
energy_label_score=170
energy_label_certificate_holder="EP Certificatie B.V."

data = {
    "Label": ["A+++", "A++", "A+", "A", "B", "C", "D", "E", "F", "G"],
    "VV": [1200000, 547000, 544000, 538000, 535000, 531000, 526000, 522000, 520000, 518000],
    "ΔVV": [53000, 16000, 13000, 0, -1000, -2000, -5000, -9000, -11000, -13000],
    "MV": [1100000, 477000, 475000, 470000, 467000, 454000, 460000, 327000, 324000, 319000],
    "ΔMV": [46000, 23000, 21000, 0, -1000, -2000, -5000, -127000, -130000, -135000],
    "WWS": [244, 239, 233, 224, 212, 195, 190, 177, 90, 80],
    "ΔWWS": [49, 44, 38, 0, -1, -3, -5, -18, -23, -28],
    "Rent": [2482, 2455, 2450, 2422, 2408, 2393, 2381, 2320, 2380, 2376],
    "ΔRent": [89, 62, 57, 0, -5, -9, -12, -113, -14, -17]
}

energy_label_impact_df = pd.DataFrame(data)



