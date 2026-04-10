import os
import numpy as np

# ((permited value),(Don’t know/ Missing))

# Columns that are interesting to analyse:
reduced_data_pd = {
    "_RFHYPE5": ((1, 2), (9)),  # Hypertension
    "TOLDHI2": (
        (1, 2),
        (9),
    ),  # told by a doctor, nurse that your blood cholesterol is high
    "_CHOLCHK": (
        (1, 2, 3),
        (9),
    ),  # Three level categorization for cholesterol checked within past five years
    "_BMI5": ((), ()),  # Calculated Body Mass Index
    "_AGEG5YR": (np.arange(1, 14), (14)),  # 18-24 and then 5-year age groupings to 80+
    "_INCOMG": (np.arange(1, 8), (9)),  # Income
    "_LTASTH1": ((1, 2), (9)),  # Chronic Health Conditions
    "_SMOKER3": ((1, 2, 3), (9)),  # Tobacco use
    "_RFDRHV5": (
        (1, 2),
        (9),
    ),  # CV for HEAVY drinking (GT 2 drinks per day for men, GT 1 drink per day for women)
    "MAXVO2_": (
        (),
        (99900),
    ),  # Estimated Age-Gender Specific Maximum Oxygen Consumption
    "_PAINDX1": ((1, 2), (9)),  # Physical Activity Index
    "_TOTINDA": ((1, 2), (9)),  # Physical activity
    "FRUTDA1_": ((), ()),  # Fruit times consumed per day
    "VEGEDA1_": ((), ()),  # Vegetable times consumed per day
    "HLTHPLN1": ((1, 2), (9)),  # Do you have any kind of health care coverage?
    "CVDSTRK3": ((1, 2), (7, 9)),  # (Ever told) you had a stroke
    "MEDCOST": (
        (1, 2),
        (7, 9),
    ),  # Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?
    "DIABETE3": (
        (1, 2, 3, 4),
        (7, 9),
    ),  # (Ever told) you have diabetes (If "Yes" and respondent is female,
    # ask "Was this only when you were pregnant?". If Respondent says pre-diabetes or borderline diabetes, use response code 4.)
    "MENTHLTH": (
        (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            88,
        ),
        (77, 99),
    ),  # for how many days during the past 30 days was your mental health not good?
    "DIFFWALK": (
        (1, 2),
        (7, 9),
    ),  # Do you have serious difficulty walking or climbing stairs?
    "SEX": ((1, 2)),  # Indicate sex of respondent.
}

cols_to_read = [
    28,
    30,
    32,
    38,
    39,
    48,
    50,
    69,
    232,
    233,
    235,
    246,
    253,
    258,
    259,
    265,
    267,
    271,
    284,
    287,
    306,
]

reduced_data = {
    28: (
        (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            88,
        ),
        (77, 99),
    ),
    30: ((1, 2), (9)),
    32: ((1, 2), (7, 9)),
    38: ((1, 2), (9)),
    39: ((1, 2), (7, 9)),
    48: ((1, 2, 3, 4), (7, 9)),
    50: ((1, 2)),
    69: ((1, 2), (7, 9)),
    232: ((1, 2), (9)),
    233: ((1, 2, 3), (9)),
    235: ((1, 2), (9)),
    246: (np.arange(1, 14), (14)),
    253: ((), ()),
    258: (np.arange(1, 8), (9)),
    259: ((1, 2, 3), (9)),
    265: ((1, 2), (9)),
    267: ((), ()),
    271: ((), ()),
    284: ((1, 2), (9)),
    287: ((), (99900)),
    306: ((1, 2), (9)),
}
