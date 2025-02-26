#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:26:04 2024

@author: koussanc
"""

import pandas as pd
import json

with open("drugcomb_drugs.txt") as json_file:
    drugcomb_drugs = json.load(json_file)
    print(drugcomb_drugs)


drugcomb_drugs_df = pd.DataFrame(drugcomb_drugs)
drugcomb_drugs_df.to_csv('drugcomb_drugs_df.csv')

